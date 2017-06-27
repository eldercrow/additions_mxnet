import tools.find_mxnet
import mxnet as mx
import logging
import sys
import os
import importlib
import re
from dataset.iterator import DetIter
from dataset.patch_iterator import PatchIter
from dataset.dataset_loader import load_pascal, load_pascal_patch
from train.metric import MultiBoxMetric
from evaluate.eval_metric import MApMetric, VOC07MApMetric
from tools.rand_sampler import RandScaler
from config.config import cfg


def convert_pretrained(name, args):
    """
    Special operations need to be made due to name inconsistance, etc

    Parameters:
    ---------
    name : str
        pretrained model name
    args : dict
        loaded arguments

    Returns:
    ---------
    processed arguments as dict
    """
    if 'vgg16_reduced' in name:
        args['conv6_bias'] = args.pop('fc6_bias')
        args['conv6_weight'] = args.pop('fc6_weight')
        args['conv7_bias'] = args.pop('fc7_bias')
        args['conv7_weight'] = args.pop('fc7_weight')
        del args['fc8_weight']
        del args['fc8_bias']
    return args

def get_lr_scheduler(learning_rate, lr_refactor_step, lr_refactor_ratio,
                     num_example, batch_size, begin_epoch):
    """
    Compute learning rate and refactor scheduler

    Parameters:
    ---------
    learning_rate : float
        original learning rate
    lr_refactor_step : comma separated str
        epochs to change learning rate
    lr_refactor_ratio : float
        lr *= ratio at certain steps
    num_example : int
        number of training images, used to estimate the iterations given epochs
    batch_size : int
        training batch size
    begin_epoch : int
        starting epoch

    Returns:
    ---------
    (learning_rate, mx.lr_scheduler) as tuple
    """
    assert lr_refactor_ratio > 0
    iter_refactor = [int(r) for r in lr_refactor_step.split(',') if r.strip()]
    if lr_refactor_ratio >= 1:
        return (learning_rate, None)
    else:
        lr = learning_rate
        epoch_size = num_example // batch_size
        for s in iter_refactor:
            if begin_epoch >= s:
                lr *= lr_refactor_ratio
        if lr != learning_rate:
            logging.getLogger().info("Adjusted learning rate to {} for epoch {}".format(lr, begin_epoch))
        steps = [epoch_size * (x - begin_epoch) for x in iter_refactor if x > begin_epoch]
        lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=lr_refactor_ratio)
        return (lr, lr_scheduler)

def train_net(net, dataset, image_set, devkit_path, batch_size,
              data_shape, mean_pixels, resume, finetune, from_scratch, pretrained, epoch,
              prefix, ctx, begin_epoch, end_epoch, frequent, learning_rate,
              momentum, weight_decay, lr_refactor_step, lr_refactor_ratio,
              year='', val_image_set=None, val_year='', freeze_layer_pattern='',
              label_pad_width=350,
              nms_thresh=0.45, force_nms=False, ovp_thresh=0.5,
              use_difficult=False, 
              voc07_metric=False, nms_topk=400, force_suppress=False,
              iter_monitor=0, monitor_pattern=".*", log_file=None):
    """
    Wrapper for training phase.

    Parameters:
    ----------
    net : str
        symbol name for the network structure
    dataset : str
        pascal_voc, imagenet...
    image_set : str
        train, trainval...
    devkit_path : str
        root directory of dataset
    batch_size : int
        training batch-size
    data_shape : int or tuple
        width/height as integer or (3, height, width) tuple
    mean_pixels : tuple of floats
        mean pixel values for red, green and blue
    resume : int
        resume from previous checkpoint if > 0
    finetune : int
        fine-tune from previous checkpoint if > 0
    pretrained : str
        prefix of pretrained model, including path
    epoch : int
        load epoch of either resume/finetune/pretrained model
    prefix : str
        prefix for saving checkpoints
    ctx : [mx.cpu()] or [mx.gpu(x)]
        list of mxnet contexts
    begin_epoch : int
        starting epoch for training, should be 0 if not otherwise specified
    end_epoch : int
        end epoch of training
    frequent : int
        frequency to print out training status
    learning_rate : float
        training learning rate
    momentum : float
        trainig momentum
    weight_decay : float
        training weight decay param
    lr_refactor_ratio : float
        multiplier for reducing learning rate
    lr_refactor_step : comma separated integers
        at which epoch to rescale learning rate, e.g. '30, 60, 90'
    year : str
        2007, 2012 or combinations splitted by comma
    freeze_layer_pattern : str
        regex pattern for layers need to be fixed
    num_example : int
        number of training images
    label_pad_width : int
        force padding training and validation labels to sync their label widths
    nms_thresh : float
        non-maximum suppression threshold for validation
    force_nms : boolean
        suppress overlaped objects from different classes
    iter_monitor : int
        monitor internal stats in networks if > 0, specified by monitor_pattern
    monitor_pattern : str
        regex pattern for monitoring network stats
    log_file : str
        log to file if enabled
    """
    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if log_file:
        fh = logging.FileHandler(log_file)
        logger.addHandler(fh)

    # check args
    if isinstance(data_shape, int):
        data_shape = (3, data_shape, data_shape)
    assert len(data_shape) == 3 and data_shape[0] == 3
    prefix += '_' + str(data_shape[1])

    if isinstance(mean_pixels, (int, float)):
        mean_pixels = [mean_pixels, mean_pixels, mean_pixels]
    assert len(mean_pixels) == 3, "must provide all RGB mean values"

    # load dataset
    if dataset == 'pascal_voc':
        imdb = load_pascal(image_set, year, devkit_path, cfg.train['init_shuffle'])
        if val_image_set and val_image_set != '' and val_year:
            val_imdb = load_pascal(val_image_set, val_year, devkit_path, False)
        else:
            val_imdb = None
    elif dataset == 'pascal_voc_patch':
        imdb = load_pascal_patch(image_set, year, devkit_path, shuffle=cfg.train['init_shuffle'], 
                patch_shape=data_shape[1])
        if val_image_set and val_image_set != '' and val_year:
            val_imdb = load_pascal_patch(val_image_set, val_year, devkit_path, shuffle=False, 
                    patch_shape=data_shape[1])
        else:
            val_imdb = None
    else:
        raise NotImplementedError("Dataset " + dataset + " not supported")

    # init iterator
    if dataset.find('_patch') < 0:
        rand_scaler = RandScaler(min_scale=cfg.train['min_aug_scale'],
                                 max_scale=cfg.train['max_aug_scale'],
                                 min_gt_scale=cfg.train['min_aug_gt_scale'],
                                 max_trials=cfg.train['max_aug_trials'],
                                 max_sample=cfg.train['max_aug_sample'],
                                 patch_size=cfg.train['aug_patch_size'])
        train_iter = DetIter(imdb, batch_size, data_shape, mean_pixels, 
                             [rand_scaler], cfg.train['rand_mirror'],
                             cfg.train['epoch_shuffle'], cfg.train['seed'],
                             is_train=True)
        # TODO: enable val_iter
        val_iter = None
        # if val_imdb:
        #     val_iter = DetIter(val_imdb, batch_size, data_shape, mean_pixels,
        #                        cfg.valid.rand_scaler, cfg.valid.rand_mirror,
        #                        cfg.valid.epoch_shuffle, cfg.valid.seed,
        #                        is_train=True)
        # else:
        #     val_iter = None
    else:
        train_iter = PatchIter(imdb, batch_size, data_shape[1], mean_pixels, is_train=True)
        val_iter = None


    # load symbol
    sys.path.append(os.path.join(cfg.ROOT_DIR, 'symbol'))
    symbol_module = importlib.import_module("symbol_" + net)
    net = symbol_module.get_symbol_train(imdb.num_classes, nms_thresh=nms_thresh,
        force_suppress=force_suppress, nms_topk=nms_topk)

    # define layers with fixed weight/bias
    if freeze_layer_pattern.strip():
        re_prog = re.compile(freeze_layer_pattern)
        fixed_param_names = [name for name in net.list_arguments() if re_prog.match(name)]
    else:
        fixed_param_names = None

    # load pretrained or resume from previous state
    ctx_str = '('+ ','.join([str(c) for c in ctx]) + ')'
    if resume > 0:
        logger.info("Resume training with {} from epoch {}"
            .format(ctx_str, resume))
        _, args, auxs = mx.model.load_checkpoint(prefix, resume)
        begin_epoch = resume
    elif finetune > 0:
        logger.info("Start finetuning with {} from epoch {}"
            .format(ctx_str, finetune))
        _, args, auxs = mx.model.load_checkpoint(prefix, finetune)
        begin_epoch = finetune
        # the prediction convolution layers name starts with relu, so it's fine
        fixed_param_names = [name for name in net.list_arguments() \
            if name.startswith('conv')]
    elif from_scratch:
        logger.info('From scratch training.')
        args = None
        auxs = None
        fixed_param_names = None
    elif pretrained:
        logger.info("Start training with {} from pretrained model {}"
            .format(ctx_str, pretrained))
        _, args, auxs = mx.model.load_checkpoint(pretrained, epoch)
        args = convert_pretrained(pretrained, args)
    else:
        logger.info("Experimental: start training from scratch with {}"
            .format(ctx_str))
        args = None
        auxs = None
        fixed_param_names = None

    # helper information
    if fixed_param_names:
        logger.info("Freezed parameters: [" + ','.join(fixed_param_names) + ']')

    # init training module
    mod = mx.mod.Module(net, label_names=('label',), logger=logger, context=ctx,
                        fixed_param_names=fixed_param_names)

    # for debug
    internals = net.get_internals()
    _, out_shapes, _ = internals.infer_shape(data=(32, 3, 256, 256), label=(32, 5))
    shape_dict = dict(zip(internals.list_outputs(), out_shapes))

    # fit parameters
    batch_end_callback = mx.callback.Speedometer(train_iter.batch_size, frequent=frequent)
    epoch_end_callback = mx.callback.do_checkpoint(prefix)
    learning_rate, lr_scheduler = get_lr_scheduler(learning_rate, lr_refactor_step,
        lr_refactor_ratio, imdb.num_images, batch_size, begin_epoch)
    optimizer_params={'learning_rate': learning_rate,
                      'wd': weight_decay,
                      'clip_gradient': 10.0,
                      'rescale_grad': 1.0,
                      'lr_scheduler': lr_scheduler}
    # optimizer_params={'learning_rate':learning_rate,
    #                   'momentum':momentum,
    #                   'wd':weight_decay,
    #                   'lr_scheduler':lr_scheduler,
    #                   'clip_gradient':None,
    #                   'rescale_grad': 1.0}
    monitor = mx.mon.Monitor(iter_monitor, pattern=monitor_pattern) if iter_monitor > 0 else None

    # run fit net, every n epochs we run evaluation network to get mAP
    # if voc07_metric:
    #     valid_metric = VOC07MApMetric(ovp_thresh, use_difficult, imdb.classes, pred_idx=3)
    # else:
    #     valid_metric = MApMetric(ovp_thresh, use_difficult, imdb.classes, pred_idx=3)
    valid_metric = None

    mod.fit(train_iter,
            val_iter,
            eval_metric=MultiBoxMetric(),
            validation_metric=valid_metric,
            batch_end_callback=batch_end_callback,
            epoch_end_callback=epoch_end_callback,
            optimizer='adam',
            optimizer_params=optimizer_params,
            begin_epoch=begin_epoch,
            num_epoch=end_epoch,
            initializer=mx.init.Xavier(),
            arg_params=args,
            aux_params=auxs,
            allow_missing=True,
            monitor=monitor)
