import tools.find_mxnet
import mxnet as mx
import logging
import sys
import os
import importlib
from initializer import ScaleInitializer
from metric import MultiBoxMetric, FaceMetric, FacePatchMetric
from dataset.iterator import DetIter
from dataset.patch_iterator import PatchIter
from dataset.pascal_voc import PascalVoc
from dataset.wider import Wider
from dataset.wider_patch import WiderPatch
from dataset.concat_db import ConcatDB
from config.config import cfg

def load_wider(image_set, devkit_path, shuffle=False, is_train=True):
    """
    wrapper function for loading wider face dataset

    Parameters:
    ----------
    image_set : str
        train, trainval...
    devkit_path : str
        root directory of dataset
    shuffle : bool
        whether to shuffle initial list

    Returns:
    ----------
    Imdb
    """
    image_set = [y.strip() for y in image_set.split(',')]
    assert image_set, "No image_set specified"

    imdbs = []
    for s in image_set:
        imdbs.append(Wider(s, devkit_path, shuffle, is_train=is_train))
    if len(imdbs) > 1:
        return ConcatDB(imdbs, shuffle)
    else:
        return imdbs[0]

def load_wider_patch(image_set, devkit_path, shuffle=False, data_shape=192):
    """
    wrapper function for loading wider face dataset

    Parameters:
    ----------
    image_set : str
        train, trainval...
    devkit_path : str
        root directory of dataset
    shuffle : bool
        whether to shuffle initial list

    Returns:
    ----------
    Imdb
    """
    image_set = [y.strip() for y in image_set.split(',')]
    assert image_set, "No image_set specified"

    imdbs = []
    for s in image_set:
        imdbs.append(WiderPatch(s, devkit_path, shuffle, is_train=True, 
            range_rand_scale=(1.0, 1.1412), patch_shape=data_shape, max_roi_size=data_shape*1.2))
    if len(imdbs) > 1:
        return ConcatDB(imdbs, shuffle)
    else:
        return imdbs[0]

# def load_pascal(image_set, year, devkit_path, shuffle=False):
#     """
#     wrapper function for loading pascal voc dataset
#
#     Parameters:
#     ----------
#     image_set : str
#         train, trainval...
#     year : str
#         2007, 2012 or combinations splitted by comma
#     devkit_path : str
#         root directory of dataset
#     shuffle : bool
#         whether to shuffle initial list
#
#     Returns:
#     ----------
#     Imdb
#     """
#     image_set = [y.strip() for y in image_set.split(',')]
#     assert image_set, "No image_set specified"
#     year = [y.strip() for y in year.split(',')]
#     assert year, "No year specified"
#
#     # make sure (# sets == # years)
#     if len(image_set) > 1 and len(year) == 1:
#         year = year * len(image_set)
#     if len(image_set) == 1 and len(year) > 1:
#         image_set = image_set * len(year)
#     assert len(image_set) == len(year), "Number of sets and year mismatch"
#
#     imdbs = []
#     for s, y in zip(image_set, year):
#         imdbs.append(PascalVoc(s, y, devkit_path, shuffle, is_train=True))
#     if len(imdbs) > 1:
#         return ConcatDB(imdbs, shuffle)
#     else:
#         return imdbs[0]
#
# def convert_pretrained(name, args):
#     """
#     Special operations need to be made due to name inconsistance, etc
#
#     Parameters:
#     ---------
#     args : dict
#         loaded arguments
#
#     Returns:
#     ---------
#     processed arguments as dict
#     """
#     if name == 'vgg16_reduced':
#         args['conv6_bias'] = args.pop('fc6_bias')
#         args['conv6_weight'] = args.pop('fc6_weight')
#         args['conv7_bias'] = args.pop('fc7_bias')
#         args['conv7_weight'] = args.pop('fc7_weight')
#         del args['fc8_weight']
#         del args['fc8_bias']
#     return args

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
               data_shape, mean_pixels, resume, finetune, pretrained, from_scratch, epoch, 
               prefix, ctx, begin_epoch, end_epoch, frequent, learning_rate,
               momentum, weight_decay, val_set, 
               lr_refactor_step, lr_refactor_ratio, 
               iter_monitor=0, log_file=None):
    """
    Wrapper for training module

    Parameters:
    ---------
    net : mx.Symbol
        training network
    dataset : str
        pascal, imagenet...
    image_set : str
        train, trainval...
    # year : str
    #     2007, 2012 or combinations splitted by comma
    devkit_path : str
        root directory of dataset
    batch_size : int
        training batch size
    data_shape : int or (int, int)
        resize image size
    mean_pixels : tuple (float, float, float)
        mean pixel values in (R, G, B)
    resume : int
        if > 0, will load trained epoch with name given by prefix
    finetune : int
        if > 0, will load trained epoch with name given by prefix, in this mode
        all convolutional layers except the last(prediction layer) are fixed
    pretrained : str
        prefix of pretrained model name
    epoch : int
        epoch of pretrained model
    prefix : str
        prefix of new model
    ctx : mx.gpu(?) or list of mx.gpu(?)
        training context
    begin_epoch : int
        begin epoch, default should be 0
    end_epoch : int
        when to stop training
    frequent : int
        frequency to log out batch_end_callback
    learning_rate : float
        learning rate, will be divided by batch_size automatically
    momentum : float
        (0, 1), training momentum
    weight_decay : float
        decay weights regardless of gradient
    val_set : str
        similar to image_set, used for validation
    # val_year : str
    #     similar to year, used for validation
    lr_refactor_epoch : int
        number of epoch to change learning rate
    lr_refactor_ratio : float
        new_lr = old_lr * lr_refactor_ratio
    iter_monitor : int
        if larger than 0, will print weights/gradients every iter_monitor iters
    log_file : str
        log to file if not None

    Returns:
    ---------
    None
    """
    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if log_file:
        fh = logging.FileHandler(log_file)
        logger.addHandler(fh)

    # kvstore
    kv = mx.kvstore.create("device")

    # check args
    if isinstance(data_shape, int):
        data_shape = (data_shape, data_shape)
    assert len(data_shape) == 2, "data_shape must be (h, w) tuple or list or int"
    prefix += '_' + str(data_shape[0])

    if isinstance(mean_pixels, (int, float)):
        mean_pixels = [mean_pixels, mean_pixels, mean_pixels]
    assert len(mean_pixels) == 3, "must provide all RGB mean values"

    # load dataset
    # if dataset == 'pascal':
    #     imdb = load_pascal(image_set, year, devkit_path, cfg.TRAIN.INIT_SHUFFLE)
    #     if val_set and val_year:
    #         val_imdb = load_pascal(val_set, val_year, devkit_path, False)
    #     else:
    #         val_imdb = None
    if dataset == 'wider':
        imdb = load_wider(image_set, devkit_path, cfg.TRAIN.INIT_SHUFFLE, is_train=True)
        if val_set:
            val_imdb = load_wider(val_set, devkit_path, False, is_train=True)
        else:
            val_imdb = None
        # init data iterator
        train_iter = DetIter(imdb, batch_size, data_shape, mean_pixels, 
                             cfg.TRAIN.RAND_SAMPLERS, cfg.TRAIN.RAND_MIRROR,
                             cfg.TRAIN.EPOCH_SHUFFLE, cfg.TRAIN.RAND_SEED,
                             is_train=True)
        if val_imdb:
            val_iter = DetIter(val_imdb, batch_size, data_shape, mean_pixels,
                               cfg.VALID.RAND_SAMPLERS, cfg.VALID.RAND_MIRROR,
                               cfg.VALID.EPOCH_SHUFFLE, cfg.VALID.RAND_SEED,
                               is_train=True)
        else:
            val_iter = None
    elif dataset == 'wider_patch':
        imdb = load_wider_patch(image_set, devkit_path, cfg.TRAIN.INIT_SHUFFLE, data_shape[0])
        if val_set:
            val_imdb = load_wider_patch(val_set, devkit_path, False, data_shape[0])
        else:
            val_imdb = None
        # init data iterator
        train_iter = PatchIter(imdb, batch_size, data_shape, mean_pixels, 
                             cfg.TRAIN.RAND_SAMPLERS, cfg.TRAIN.RAND_MIRROR,
                             cfg.TRAIN.EPOCH_SHUFFLE, cfg.TRAIN.RAND_SEED,
                             is_train=True)
        if val_imdb:
            val_iter = PatchIter(val_imdb, batch_size, data_shape, mean_pixels,
                               cfg.VALID.RAND_SAMPLERS, cfg.VALID.RAND_MIRROR,
                               cfg.VALID.EPOCH_SHUFFLE, cfg.VALID.RAND_SEED,
                               is_train=True)
        else:
            val_iter = None
    else:
        raise NotImplementedError("Dataset " + dataset + " not supported")

    # load symbol
    sys.path.append(os.path.join(cfg.ROOT_DIR, 'symbol'))
    if dataset == 'wider':
        net = importlib.import_module("symbol_" + net).get_symbol_train(\
                imdb.num_classes) #, n_group=8, patch_size=768)
        # freeze bn layers
        fixed_param_names = None # [name for name in net.list_arguments() if name.endswith('_gamma')]
        eval_metric = FacePatchMetric()
    elif dataset == 'wider_patch':
        net = importlib.import_module("symbol_" + net).get_symbol_train(\
                imdb.num_classes) #, n_group=6, patch_size=256)
        eval_metric = FacePatchMetric()
    #
    # # define layers with fixed weight/bias
    # fixed_param_names = [name for name in net.list_arguments() \
    #     if name.startswith('conv1_') or name.startswith('conv2_') or name.endswith('_gamma')]

    # init training module
    mod = None

    # load pretrained or resume from previous state
    ctx_str = '('+ ','.join([str(c) for c in ctx]) + ')'
    if resume > 0:
        logger.info("Resume training with {} from epoch {}"
            .format(ctx_str, resume))
        mod = mx.mod.Module.load(prefix, resume, load_optimizer_states=True,
                label_names=[('label')], logger=logger, context=ctx)
        args = None
        auxs = None
        # _, args, auxs = mx.model.load_checkpoint(prefix, resume)
        # import ipdb
        # ipdb.set_trace()
        # auxs.pop('multibox_target_target_loc_weight')
        begin_epoch = resume
        fixed_param_names = None
    elif finetune > 0:
        logger.info("Start finetuning with {} from epoch {}"
            .format(ctx_str, finetune))
        _, args, auxs = mx.model.load_checkpoint(prefix, finetune)
        begin_epoch = finetune
        # the prediction convolution layers name starts with relu, so it's fine
        # fixed_param_names = [name for name in net.list_arguments() \
        #     if name.startswith('conv')]
    elif from_scratch:
        logger.info("Experimental: start training from scratch with {}"
            .format(ctx_str))
        args = None
        auxs = None
        fixed_param_names = None
    elif pretrained:
        logger.info("Start training with {} from pretrained model {}"
            .format(ctx_str, pretrained))
        _, args, auxs = mx.model.load_checkpoint(pretrained, epoch)
        # del auxs['multibox_target_mean_pos_prob_bias']
        # del auxs['multibox_target_target_loc_weight']
        # args = convert_pretrained(pretrained, args)
    else:
        logger.info("Experimental: start training from scratch with {}"
            .format(ctx_str))
        args = None
        auxs = None
        fixed_param_names = None

    # helper information
    if fixed_param_names:
        logger.info("Freezed parameters: [" + ','.join(fixed_param_names) + ']')

    if mod is None:
        mod = mx.mod.Module(net, label_names=('label',), logger=logger, context=ctx,
                            fixed_param_names=fixed_param_names)
    # fit
    batch_end_callback = mx.callback.Speedometer(train_iter.batch_size, frequent=frequent, auto_reset=False)
    epoch_end_callback = mx.callback.module_checkpoint(mod, prefix, 1, True)
    num_example=imdb.num_images
    learning_rate, lr_scheduler = get_lr_scheduler(learning_rate, lr_refactor_step,
            lr_refactor_ratio, num_example, batch_size, begin_epoch)
    optimizer_params={'learning_rate': learning_rate,
                      'wd': weight_decay,
                      'lr_scheduler': lr_scheduler, 
                      'clip_gradient': -1.0,
                      'rescale_grad': 1.0}
    # optimizer_params={'learning_rate':learning_rate,
    #                   'momentum':momentum,
    #                   'wd':weight_decay,
    #                   'lr_scheduler':lr_scheduler,
    #                   'clip_gradient':4.0,
    #                   'rescale_grad': 1.0}
    monitor_pattern = '.*weight|.*bias|.*beta|.*gamma|.*moving_mean|.*moving_var'
    monitor = mx.mon.Monitor(iter_monitor, pattern=monitor_pattern) if iter_monitor > 0 else None
    initializer = mx.init.Mixed([".*scale", ".*"], \
        [ScaleInitializer(), mx.init.Xavier(magnitude=2.34)])

    mod.fit(train_iter,
            eval_data=val_iter,
            eval_metric=eval_metric, # MultiBoxMetric(),
            batch_end_callback=batch_end_callback,
            epoch_end_callback=epoch_end_callback,
            optimizer='rmsprop',
            optimizer_params=optimizer_params,
            kvstore = kv,
            begin_epoch=begin_epoch,
            num_epoch=end_epoch,
            initializer=initializer,
            arg_params=args,
            aux_params=auxs,
            allow_missing=True,
            monitor=monitor)
