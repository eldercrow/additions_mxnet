import tools.find_mxnet
import mxnet as mx
import logging
import sys
import os
import importlib
import re
from dataset.iterator import DetIter
from dataset.dataset_loader import load_pascal, load_wider #, load_pascal_patch
from train.metric import MultiBoxMetric #, FacePatchMetric, RONMetric
from train.train_net_common import train_net_common
from tools.rand_sampler import RandScaler
from plateau_lr import PlateauScheduler
from plateau_module import PlateauModule
from config.config import cfg
from symbol.symbol_factory import get_symbol_train


def train_net(net, dataset, image_set, devkit_path, batch_size,
              data_shape, mean_pixels, resume, finetune, from_scratch, pretrained, epoch,
              prefix, ctx, begin_epoch, end_epoch, frequent,
              optimizer_name='adam', learning_rate=1e-03, momentum=0.9, weight_decay=5e-04,
              lr_refactor_step=(3,4,5,6), lr_refactor_ratio=0.1,
              val_image_set='', val_year='',
              use_plateau=True,
              year='', freeze_layer_pattern='',
              force_resize=True,
              min_obj_size=32.0, use_difficult=False,
              nms_thresh=0.45, force_suppress=False, ovp_thresh=0.5,
              voc07_metric=True, nms_topk=400,
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

    if isinstance(mean_pixels, (int, float)):
        mean_pixels = [mean_pixels, mean_pixels, mean_pixels]
    assert len(mean_pixels) == 3, "must provide all RGB mean values"

    # load dataset
    val_imdb = None
    if dataset == 'pascal_voc':
        imdb = load_pascal(image_set, year, devkit_path, cfg.train['shuffle'])
        if val_image_set:
            assert val_year
            val_imdb = load_pascal(val_image_set, val_year, devkit_path, False)
            max_objects = max(imdb.max_objects, val_imdb.max_objects)
            imdb.pad_labels(max_objects)
            val_imdb.pad_labels(max_objects)
        force_resize = True
    elif dataset == 'wider':
        imdb = load_wider(image_set, devkit_path, cfg.train['shuffle'])
        force_resize = False
    elif dataset == 'mscoco':
        imdb = load_mscoco(image_set, devkit_path, cfg.train['shuffle'])
        force_resize = True
    else:
        raise NotImplementedError("Dataset " + dataset + " not supported")

    # init iterator
    patch_size = data_shape[1]
    min_gt_scale = min_obj_size / float(patch_size)
    rand_scaler = RandScaler(patch_size, min_gt_scale=min_gt_scale, force_resize=force_resize)
    train_iter = DetIter(imdb, batch_size, data_shape[1], rand_scaler,
                         mean_pixels, cfg.train['rand_mirror_prob'] > 0,
                         cfg.train['shuffle'], cfg.train['seed'],
                         is_train=True)
    if val_imdb:
        rand_scaler = RandScaler(patch_size, no_random=True, force_resize=force_resize)
        val_iter = DetIter(val_imdb, batch_size, data_shape[1], rand_scaler,
                           mean_pixels, is_train=True)
    else:
        val_iter = None

    train_net_common(net, train_iter, val_iter, batch_size,
                     data_shape, resume, finetune, pretrained, epoch,
                     prefix, ctx, begin_epoch, end_epoch, frequent, learning_rate,
                     momentum, weight_decay, use_plateau, lr_refactor_step, lr_refactor_ratio,
                     freeze_layer_pattern, imdb.num_images, imdb.max_objects,
                     nms_thresh, force_suppress, ovp_thresh,
                     use_difficult, imdb.classes,
                     optimizer_name,
                     voc07_metric, nms_topk,
                     iter_monitor,
                     monitor_pattern, logger)

    # # load symbol
    # net = get_symbol_train(net, data_shape[1], num_classes=imdb.num_classes)
    #     # nms_thresh=0.45, force_suppress=False, nms_topk=400)
    #
    # # define layers with fixed weight/bias
    # if freeze_layer_pattern.strip():
    #     re_prog = re.compile(freeze_layer_pattern)
    #     fixed_param_names = [name for name in net.list_arguments() if re_prog.match(name)]
    # else:
    #     fixed_param_names = None
    #
    # # load pretrained or resume from previous state
    # ctx_str = '('+ ','.join([str(c) for c in ctx]) + ')'
    # if resume > 0:
    #     logger.info("Resume training with {} from epoch {}"
    #         .format(ctx_str, resume))
    #     _, args, auxs = mx.model.load_checkpoint(prefix, resume)
    #     begin_epoch = resume
    # elif pretrained:
    #     try:
    #         logger.info("Start training with {} from pretrained model {}"
    #             .format(ctx_str, pretrained))
    #         _, args, auxs = mx.model.load_checkpoint(pretrained, epoch)
    #     except:
    #         logger.info("Failed to load the pretrained model. Start from scratch.")
    #         args = None
    #         auxs = None
    #         fixed_param_names = None
    # else:
    #     logger.info("Experimental: start training from scratch with {}"
    #         .format(ctx_str))
    #     args = None
    #     auxs = None
    #     fixed_param_names = None
    #
    # # helper information
    # if fixed_param_names:
    #     logger.info("Freezed parameters: [" + ','.join(fixed_param_names) + ']')
    #
    # # init training module
    # if not use_plateau: #cfg.train['use_focal_loss']: # focal loss does not go well with plateau
    #     mod = mx.mod.Module(net, label_names=('label',), logger=logger, context=ctx,
    #             fixed_param_names=fixed_param_names)
    # else:
    #     mod = PlateauModule(net, label_names=('label',), logger=logger, context=ctx,
    #             fixed_param_names=fixed_param_names)
    #
    # # robust parameter setting
    # mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
    # mod = set_mod_params(mod, args, auxs, data_shape, logger)
    #
    # # fit parameters
    # batch_end_callback = mx.callback.Speedometer(train_iter.batch_size, frequent=frequent, auto_reset=True)
    # epoch_end_callback = mx.callback.do_checkpoint(prefix)
    # monitor = mx.mon.Monitor(iter_monitor, pattern=monitor_pattern) if iter_monitor > 0 else None
    # optimizer_params={'learning_rate': learning_rate,
    #                   'wd': weight_decay,
    #                   'clip_gradient': 10.0,
    #                   'rescale_grad': 1.0}
    # if optimizer_name == 'sgd':
    #     optimizer_params['momentum'] = momentum
    #
    # if not use_plateau: #cfg.train['use_focal_loss']:
    #     learning_rate, lr_scheduler = get_lr_scheduler(learning_rate, lr_refactor_step,
    #             lr_refactor_ratio, num_example, batch_size, begin_epoch)
    # else:
    #     w_l1 = cfg.train['smoothl1_weight']
    #     eval_weights = {'CrossEntropy': 1.0, 'SmoothL1': w_l1}
    #     plateau_lr = PlateauScheduler( \
    #             patient_epochs=lr_refactor_step, factor=float(lr_refactor_ratio), eval_weights=eval_weights)
    #     plateau_metric = MultiBoxMetric()
    #
    # eval_metric = MultiBoxMetric()
    # valid_metric = None
    #
    # if not use_plateau: #cfg.train['use_focal_loss']:
    #     mod.fit(train_iter,
    #             eval_data=val_iter,
    #             eval_metric=eval_metric,
    #             validation_metric=valid_metric,
    #             batch_end_callback=batch_end_callback,
    #             epoch_end_callback=epoch_end_callback,
    #             optimizer=optimizer_name,
    #             optimizer_params=optimizer_params,
    #             begin_epoch=begin_epoch,
    #             num_epoch=end_epoch,
    #             initializer=mx.init.Xavier(),
    #             arg_params=args,
    #             aux_params=auxs,
    #             allow_missing=True,
    #             monitor=monitor)
    # else:
    #     mod.fit(train_iter,
    #             plateau_lr, plateau_metric=plateau_metric,
    #             fn_curr_model=prefix+'-1000.params',
    #             plateau_backtrace=False,
    #             eval_data=val_iter,
    #             eval_metric=eval_metric,
    #             validation_metric=valid_metric,
    #             validation_period=5,
    #             batch_end_callback=batch_end_callback,
    #             epoch_end_callback=epoch_end_callback,
    #             optimizer=optimizer_name,
    #             optimizer_params=optimizer_params,
    #             begin_epoch=begin_epoch,
    #             num_epoch=end_epoch,
    #             initializer=mx.init.Xavier(),
    #             arg_params=args,
    #             aux_params=auxs,
    #             allow_missing=True,
    #             monitor=monitor)
