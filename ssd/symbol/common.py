import mxnet as mx
import numpy as np
from layer.multibox_prior_layer import *

def convolution(data, name, num_filter, kernel, pad, stride=(1,1), dilate=(1,1), no_bias=False, lr_mult=1.0):
    ''' convolution with lr_mult and wd_mult '''
    w = mx.sym.var(name+'_weight', lr_mult=lr_mult, wd_mult=lr_mult)
    b = None
    if no_bias == False:
        b = mx.sym.var(name+'_bias', lr_mult=lr_mult*2.0, wd_mult=0.0)
    conv = mx.sym.Convolution(data, weight=w, bias=b, name=name, num_filter=num_filter,
            kernel=kernel, pad=pad, stride=stride, dilate=dilate, no_bias=no_bias)
    return conv


def fullyconnected(data, name, num_hidden, no_bias=False, lr_mult=1.0):
    ''' convolution with lr_mult and wd_mult '''
    w = mx.sym.var(name+'_weight', lr_mult=lr_mult, wd_mult=lr_mult)
    b = None
    if no_bias == False:
        b = mx.sym.var(name+'_bias', lr_mult=lr_mult*2.0, wd_mult=0.0)
    fc = mx.sym.FullyConnected(data, weight=w, bias=b, name=name, num_hidden=num_hidden, no_bias=no_bias)
    return fc


def batchnorm(data, name, use_global_stats, fix_gamma=False, lr_mult=1.0):
    ''' batch norm with lr_mult and wd_mult '''
    g = mx.sym.var(name+'_gamma', lr_mult=lr_mult, wd_mult=0.0)
    b = mx.sym.var(name+'_beta', lr_mult=lr_mult, wd_mult=0.0)
    bn = mx.sym.BatchNorm(data, gamma=g, beta=b, name=name,
            use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    return bn


def pool(data, name=None, kernel=(2, 2), stride=(2, 2), pool_type='max'):
    return mx.sym.Pooling(data, name=name, kernel=kernel, stride=stride, pool_type=pool_type)


def subpixel_upsample(data, ch, c, r):
    if r == 1 and c == 1:
        return data
    X = mx.sym.reshape(data=data, shape=(-3, 0, 0))  # (bsize*ch*r*c, a, b)
    X = mx.sym.reshape(
        data=X, shape=(-4, -1, r * c, 0, 0))  # (bsize*ch, r*c, a, b)
    X = mx.sym.transpose(data=X, axes=(0, 3, 2, 1))  # (bsize*ch, b, a, r*c)
    X = mx.sym.reshape(data=X, shape=(0, 0, -1, c))  # (bsize*ch, b, a*r, c)
    X = mx.sym.transpose(data=X, axes=(0, 2, 1, 3))  # (bsize*ch, a*r, b, c)
    X = mx.sym.reshape(
        data=X, shape=(-4, -1, ch, 0, -3))  # (bsize, ch, a*r, b*c)
    return X


def conv_act_layer(from_layer, name, num_filter, kernel=(1,1), pad=(0,0), \
    stride=(1,1), act_type="relu", use_batchnorm=False):
    """
    wrapper for a small Convolution group

    Parameters:
    ----------
    from_layer : mx.symbol
        continue on which layer
    name : str
        base name of the new layers
    num_filter : int
        how many filters to use in Convolution layer
    kernel : tuple (int, int)
        kernel size (h, w)
    pad : tuple (int, int)
        padding size (h, w)
    stride : tuple (int, int)
        stride size (h, w)
    act_type : str
        activation type, can be relu...
    use_batchnorm : bool
        whether to use batch normalization

    Returns:
    ----------
    (conv, relu) mx.Symbols
    """
    assert not use_batchnorm, "batchnorm not yet supported"
    bias = mx.symbol.Variable(name="conv{}_bias".format(name),
        init=mx.init.Constant(0.0), attr={'__lr_mult__': '2.0'})
    conv = mx.symbol.Convolution(data=from_layer, bias=bias, kernel=kernel, pad=pad, \
        stride=stride, num_filter=num_filter, name="conv{}".format(name))
    relu = mx.symbol.Activation(data=conv, act_type=act_type, \
        name="{}{}".format(act_type, name))
    if use_batchnorm:
        relu = mx.symbol.BatchNorm(data=relu, name="bn{}".format(name))
    return conv, relu

def multibox_layer(from_layers, num_classes, sizes=[.2, .95],
                    ratios=[1], normalization=-1, num_channels=[],
                    clip=True, interm_layer=0, steps=[], has_bg_class=False):
    """
    the basic aggregation module for SSD detection. Takes in multiple layers,
    generate multiple object detection targets by customized layers

    Parameters:
    ----------
    from_layers : list of mx.symbol
        generate multibox detection from layers
    num_classes : int
        number of classes excluding background, will automatically handle
        background in this function
    sizes : list or list of list
        [min_size, max_size] for all layers or [[], [], []...] for specific layers
    ratios : list or list of list
        [ratio1, ratio2...] for all layers or [[], [], ...] for specific layers
    normalizations : int or list of int
        use normalizations value for all layers or [...] for specific layers,
        -1 indicate no normalizations and scales
    num_channels : list of int
        number of input layer channels, used when normalization is enabled, the
        length of list should equals to number of normalization layers
    clip : bool
        whether to clip out-of-image boxes
    interm_layer : int
        if > 0, will add a intermediate Convolution layer
    steps : list
        specify steps for each MultiBoxPrior layer, leave empty, it will calculate
        according to layer dimensions

    Returns:
    ----------
    list of outputs, as [loc_preds, cls_preds, anchor_boxes]
    loc_preds : localization regression prediction
    cls_preds : classification prediction
    anchor_boxes : generated anchor boxes
    """
    assert len(from_layers) > 0, "from_layers must not be empty list"
    assert num_classes > 0, \
        "num_classes {} must be larger than 0".format(num_classes)

    assert len(ratios) > 0, "aspect ratios must not be empty list"
    if not isinstance(ratios[0], list):
        # provided only one ratio list, broadcast to all from_layers
        ratios = [ratios] * len(from_layers)
    assert len(ratios) == len(from_layers), \
        "ratios and from_layers must have same length"

    assert len(sizes) > 0, "sizes must not be empty list"
    if len(sizes) == 2 and not isinstance(sizes[0], list):
        # provided size range, we need to compute the sizes for each layer
         assert sizes[0] > 0 and sizes[0] < 1
         assert sizes[1] > 0 and sizes[1] < 1 and sizes[1] > sizes[0]
         tmp = np.linspace(sizes[0], sizes[1], num=(len(from_layers)-1))
         min_sizes = [start_offset] + tmp.tolist()
         max_sizes = tmp.tolist() + [tmp[-1]+start_offset]
         sizes = zip(min_sizes, max_sizes)
    assert len(sizes) == len(from_layers), \
        "sizes and from_layers must have same length"

    if not isinstance(normalization, list):
        normalization = [normalization] * len(from_layers)
    assert len(normalization) == len(from_layers)

    assert sum(x > 0 for x in normalization) == len(num_channels), \
        "must provide number of channels for each normalized layer"

    if steps:
        assert len(steps) == len(from_layers), "provide steps for all layers or leave empty"

    loc_pred_layers = []
    cls_pred_layers = []
    anchor_layers = []
    if not has_bg_class:
        num_classes += 1 # always use background as label 0

    for k, from_layer in enumerate(from_layers):
        from_name = from_layer.name
        # normalize
        if normalization[k] > 0:
            from_layer = mx.symbol.L2Normalization(data=from_layer, \
                mode="channel", name="{}_norm".format(from_name))
            scale = mx.symbol.Variable(name="{}_scale".format(from_name),
                shape=(1, num_channels.pop(0), 1, 1),
                init=mx.init.Constant(normalization[k]))
            from_layer = mx.symbol.broadcast_mul(lhs=scale, rhs=from_layer)
        if interm_layer > 0:
            from_layer = mx.symbol.Convolution(data=from_layer, kernel=(3,3), \
                stride=(1,1), pad=(1,1), num_filter=interm_layer, \
                name="{}_inter_conv".format(from_name))
            from_layer = mx.symbol.Activation(data=from_layer, act_type="relu", \
                name="{}_inter_relu".format(from_name))

        # estimate number of anchors per location
        # here I follow the original version in caffe
        # TODO: better way to shape the anchors??
        size = sizes[k]
        assert len(size) > 0, "must provide at least one size"
        size_str = "(" + ",".join([str(x) for x in size]) + ")"
        ratio = ratios[k]
        assert len(ratio) > 0, "must provide at least one ratio"
        ratio_str = "(" + ",".join([str(x) for x in ratio]) + ")"
        num_anchors = len(size) * len(ratio)
        # num_anchors = len(size) -1 + len(ratio)

        # create location prediction layer
        num_loc_pred = num_anchors * 4
        bias = mx.symbol.Variable(name="{}_loc_pred_conv_bias".format(from_name),
            init=mx.init.Constant(0.0), attr={'__lr_mult__': '2.0'})
        loc_pred = mx.symbol.Convolution(data=from_layer, bias=bias, kernel=(3,3), \
            stride=(1,1), pad=(1,1), num_filter=num_loc_pred, \
            name="{}_loc_pred_conv".format(from_name))
        loc_pred = mx.symbol.transpose(loc_pred, axes=(0,2,3,1))
        loc_pred = mx.symbol.Flatten(data=loc_pred)
        loc_pred_layers.append(loc_pred)

        # create class prediction layer
        num_cls_pred = num_anchors * num_classes
        bias = mx.symbol.Variable(name="{}_cls_pred_conv_bias".format(from_name),
            init=mx.init.Constant(0.0), attr={'__lr_mult__': '2.0'})
        cls_pred = mx.symbol.Convolution(data=from_layer, bias=bias, kernel=(3,3), \
            stride=(1,1), pad=(1,1), num_filter=num_cls_pred, \
            name="{}_cls_pred_conv".format(from_name))
        cls_pred = mx.symbol.transpose(cls_pred, axes=(0,2,3,1))
        cls_pred = mx.symbol.Flatten(data=cls_pred)
        cls_pred_layers.append(cls_pred)

        # create anchor generation layer
        if steps:
            step = (steps[k], steps[k])
        else:
            step = '(-1.0, -1.0)'
        anchors = mx.contrib.symbol.MultiBoxPrior(from_layer, sizes=size_str, ratios=ratio_str, \
            clip=clip, name="{}_anchors".format(from_name), steps=step)
        anchors = mx.symbol.Flatten(data=anchors)
        anchor_layers.append(anchors)

    loc_preds = mx.symbol.Concat(*loc_pred_layers, num_args=len(loc_pred_layers), \
        dim=1, name="multibox_loc_pred")
    cls_preds = mx.symbol.Concat(*cls_pred_layers, num_args=len(cls_pred_layers), \
        dim=1)
    cls_preds = mx.symbol.Reshape(data=cls_preds, shape=(0, -1, num_classes))
    cls_preds = mx.symbol.transpose(cls_preds, axes=(0, 2, 1), name="multibox_cls_pred")
    anchor_boxes = mx.symbol.Concat(*anchor_layers, \
        num_args=len(anchor_layers), dim=1)
    anchor_boxes = mx.symbol.Reshape(data=anchor_boxes, shape=(0, -1, 4), name="multibox_anchors")
    return [loc_preds, cls_preds, anchor_boxes]


def multibox_layer_python(from_layers, num_classes, sizes, ratios, strides, per_cls_reg=False, clip=False):
    ''' multibox layer '''
    # parameter check
    assert len(from_layers) > 0, "from_layers must not be empty list"
    assert num_classes > 1, "num_classes {} must be larger than 1".format(num_classes)
    assert len(ratios) == len(from_layers), "ratios and from_layers must have same length"
    assert len(sizes) == len(from_layers), "sizes and from_layers must have same length"

    pred_layers = []
    anchor_layers = []

    num_reg = 4 * num_classes if per_cls_reg else 4

    for k, from_layer in enumerate(from_layers):
        from_name = from_layer.name
        num_anchors = len(sizes[k]) * len(ratios[k])
        num_loc_pred = num_anchors * num_reg
        num_cls_pred = num_anchors * num_classes
        num_filter = num_loc_pred + num_cls_pred

        pred_conv = convolution(from_layer, name='{}_pred/conv'.format(from_name),
                num_filter=num_filter, kernel=(3, 3), pad=(1, 1))
        pred_conv = mx.sym.transpose(pred_conv, axes=(0, 2, 3, 1))  # (n h w ac), a=num_anchors
        pred_conv = mx.sym.reshape(pred_conv, shape=(0, -1, num_classes + num_reg))
        # pred_conv = mx.sym.reshape(pred_conv, shape=(0, -3, -4, num_anchors, -1))  # (n h*w a c)
        # pred_conv = mx.sym.reshape(pred_conv, shape=(0, -3, -1))  # (n h*w*a c)
        pred_layers.append(pred_conv)

    anchors = mx.sym.Custom(*from_layers, op_type='multibox_prior_python',
        sizes=sizes, ratios=ratios, strides=strides, clip=int(clip))
    preds = mx.sym.concat(*pred_layers, num_args=len(pred_layers), dim=1)
    return [preds, anchors]
