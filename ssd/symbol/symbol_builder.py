import mxnet as mx
from common import multi_layer_feature, multibox_layer
from layer.multibox_target_layer import *
from layer.dummy_layer import *
from layer.reweight_loss_layer import *
from config.config import cfg


def import_module(module_name):
    """Helper function to import module"""
    import sys, os
    import importlib
    sys.path.append(os.path.dirname(__file__))
    return importlib.import_module(module_name)

def get_symbol_train(network, num_classes, from_layers, num_filters, strides, pads,
                     sizes, ratios, normalizations=-1, steps=[], min_filter=128,
                     nms_thresh=0.5, force_suppress=False, nms_topk=400, **kwargs):
    """Build network symbol for training SSD

    Parameters
    ----------
    network : str
        base network symbol name
    num_classes : int
        number of object classes not including background
    from_layers : list of str
        feature extraction layers, use '' for add extra layers
        For example:
        from_layers = ['relu4_3', 'fc7', '', '', '', '']
        which means extract feature from relu4_3 and fc7, adding 4 extra layers
        on top of fc7
    num_filters : list of int
        number of filters for extra layers, you can use -1 for extracted features,
        however, if normalization and scale is applied, the number of filter for
        that layer must be provided.
        For example:
        num_filters = [512, -1, 512, 256, 256, 256]
    strides : list of int
        strides for the 3x3 convolution appended, -1 can be used for extracted
        feature layers
    pads : list of int
        paddings for the 3x3 convolution, -1 can be used for extracted layers
    sizes : list or list of list
        [min_size, max_size] for all layers or [[], [], []...] for specific layers
    ratios : list or list of list
        [ratio1, ratio2...] for all layers or [[], [], ...] for specific layers
    normalizations : int or list of int
        use normalizations value for all layers or [...] for specific layers,
        -1 indicate no normalizations and scales
    steps : list
        specify steps for each MultiBoxPrior layer, leave empty, it will calculate
        according to layer dimensions
    min_filter : int
        minimum number of filters used in 1x1 convolution
    nms_thresh : float
        non-maximum suppression threshold
    force_suppress : boolean
        whether suppress different class objects
    nms_topk : int
        apply NMS to top K detections

    Returns
    -------
    mx.Symbol

    """
    use_python_layer = True
    use_focal_loss = cfg.train['use_focal_loss']

    label = mx.sym.Variable('label')
    kwargs['use_global_stats'] = False

    if network == 'fasterface':
        loc_preds, cls_preds, anchor_boxes = import_module(network).get_symbol( \
                num_classes, sizes=sizes, ratios=ratios, steps=steps, **kwargs)
    else:
        body = import_module(network).get_symbol(num_classes, **kwargs)
        layers = multi_layer_feature(body, from_layers, num_filters, strides, pads,
            min_filter=min_filter)

        loc_preds, cls_preds, anchor_boxes = multibox_layer(layers, \
            num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
            num_channels=num_filters, clip=False, interm_layer=0, steps=steps)

    if use_python_layer:
        neg_ratio = -1 if use_focal_loss else 3
        th_small = 0.04 if not 'th_small' in kwargs else kwargs['th_small']
        cls_probs = mx.sym.SoftmaxActivation(cls_preds, mode='channel')
        tmp = mx.sym.Custom(*[anchor_boxes, label, cls_probs], name='multibox_target',
                op_type='multibox_target', hard_neg_ratio=neg_ratio, th_small=th_small)
    else:
        neg_ratio = -1 if use_focal_loss else 3
        tmp = mx.contrib.symbol.MultiBoxTarget(
            *[anchor_boxes, label, cls_preds], overlap_threshold=.5, \
            ignore_label=-1, negative_mining_ratio=neg_ratio, minimum_negative_samples=0, \
            negative_mining_thresh=.5, variances=(0.1, 0.1, 0.2, 0.2),
            name="multibox_target")
    loc_target = tmp[0]
    loc_target_mask = tmp[1]
    cls_target = tmp[2]

    if use_focal_loss:
        cls_loss = mx.symbol.SoftmaxOutput(data=cls_preds, label=cls_target, \
            ignore_label=-1, use_ignore=True, grad_scale=1., multi_output=True, \
            normalization='null', name="cls_prob", out_grad=True)
        cls_loss = mx.sym.Custom(cls_loss, cls_target, op_type='reweight_loss', name='focal_loss',
                gamma=2.0, alpha=0.25, normalize=True)
        cls_loss = mx.sym.MakeLoss(cls_loss, grad_scale=1.0, name='cls_loss')
    else:
        cls_loss = mx.symbol.SoftmaxOutput(data=cls_preds, label=cls_target, \
            ignore_label=-1, use_ignore=True, grad_scale=1., multi_output=True, \
            normalization='valid', name="cls_loss")
    loc_loss_ = mx.symbol.smooth_l1(name="loc_loss_", \
        data=loc_target_mask * (loc_preds - loc_target), scalar=1.0)
    loc_loss = mx.symbol.MakeLoss(loc_loss_, grad_scale=1.0, \
        normalization='valid', name="loc_loss")

    # monitoring training status
    cls_label = mx.sym.BlockGrad(cls_target, name="cls_label")
    loc_label = mx.sym.BlockGrad(loc_target_mask, name='loc_label')
    det = mx.contrib.symbol.MultiBoxDetection(*[cls_loss, loc_preds, anchor_boxes], \
        name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress,
        variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)
    det = mx.symbol.MakeLoss(data=det, grad_scale=0, name="det_out")

    # group output
    out = mx.symbol.Group([cls_loss, loc_loss, cls_label, loc_label, det])
    return out

def get_symbol(network, num_classes, from_layers, num_filters, sizes, ratios,
               strides, pads, normalizations=-1, steps=[], min_filter=128,
               nms_thresh=0.5, force_suppress=False, nms_topk=400, **kwargs):
    """Build network for testing SSD

    Parameters
    ----------
    network : str
        base network symbol name
    num_classes : int
        number of object classes not including background
    from_layers : list of str
        feature extraction layers, use '' for add extra layers
        For example:
        from_layers = ['relu4_3', 'fc7', '', '', '', '']
        which means extract feature from relu4_3 and fc7, adding 4 extra layers
        on top of fc7
    num_filters : list of int
        number of filters for extra layers, you can use -1 for extracted features,
        however, if normalization and scale is applied, the number of filter for
        that layer must be provided.
        For example:
        num_filters = [512, -1, 512, 256, 256, 256]
    strides : list of int
        strides for the 3x3 convolution appended, -1 can be used for extracted
        feature layers
    pads : list of int
        paddings for the 3x3 convolution, -1 can be used for extracted layers
    sizes : list or list of list
        [min_size, max_size] for all layers or [[], [], []...] for specific layers
    ratios : list or list of list
        [ratio1, ratio2...] for all layers or [[], [], ...] for specific layers
    normalizations : int or list of int
        use normalizations value for all layers or [...] for specific layers,
        -1 indicate no normalizations and scales
    steps : list
        specify steps for each MultiBoxPrior layer, leave empty, it will calculate
        according to layer dimensions
    min_filter : int
        minimum number of filters used in 1x1 convolution
    nms_thresh : float
        non-maximum suppression threshold
    force_suppress : boolean
        whether suppress different class objects
    nms_topk : int
        apply NMS to top K detections

    Returns
    -------
    mx.Symbol

    """
    kwargs['use_global_stats'] = True
    if network == 'fasterface':
        loc_preds, cls_preds, anchor_boxes = import_module(network).get_symbol( \
                num_classes, sizes=sizes, ratios=ratios, steps=steps, **kwargs)
    else:
        body = import_module(network).get_symbol(num_classes, **kwargs)
        layers = multi_layer_feature(body, from_layers, num_filters, strides, pads,
            min_filter=min_filter)

        loc_preds, cls_preds, anchor_boxes = multibox_layer(layers, \
            num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
            num_channels=num_filters, clip=False, interm_layer=0, steps=steps)
    # body = import_module(network).get_symbol(num_classes, **kwargs)
    # layers = multi_layer_feature(body, from_layers, num_filters, strides, pads,
    #     min_filter=min_filter)
    #
    # loc_preds, cls_preds, anchor_boxes = multibox_layer(layers, \
    #     num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
    #     num_channels=num_filters, clip=False, interm_layer=0, steps=steps)

    cls_prob = mx.symbol.SoftmaxActivation(data=cls_preds, mode='channel', \
        name='cls_prob')
    out = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
        name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress,
        variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk, clip=False)
    return out
