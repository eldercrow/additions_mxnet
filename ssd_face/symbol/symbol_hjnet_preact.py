from hjnet_preact import bn_relu_conv, get_hjnet_preact
from anchor_target_layer import *
import numpy as np

def build_hyperfeature(data, ctx_data, name, num_filter_proj, num_filter_hyper, scale, use_global_stats):
    """
    """
    ctx_proj = bn_relu_conv(data=ctx_data, num_filter=num_filter_proj, prefix_name=name+'/proj/', 
            kernel=(3,3), pad=(1,1), stride=(1,1), 
            use_global_stats=use_global_stats, fix_gamma=False)
    ctx_up = mx.symbol.UpSampling(ctx_proj, num_args=1, name=name+'/up', scale=scale, sample_type='nearest')
    concat = mx.symbol.Concat(data, ctx_up)
    hyper, _ = bn_relu_conv(data=concat, num_filter=num_filter_hyper, prefix_name=name+'/conv/', 
            kernel=(1,1), 
            use_global_stats=use_global_stats, fix_gamma=True)
    return hyper

def multibox_layer(from_layers, num_classes, sizes, ratios, use_global_stats, clip=True):
    ''' multibox layer '''
    # parameter check
    assert len(from_layers) > 0, "from_layers must not be empty list"
    assert num_classes > 0, "num_classes {} must be larger than 0".format(num_classes)
    assert len(ratios) > 0, "aspect ratios must not be empty list"
    if not isinstance(ratios[0], list):
        # provided only one ratio list, broadcast to all from_layers
        ratios = [ratios] * len(from_layers)
    assert len(ratios) == len(from_layers), "ratios and from_layers must have same length"
    assert len(sizes) > 0, "sizes must not be empty list"
    if len(sizes) == 2 and not isinstance(sizes[0], list):
        # provided size range, we need to compute the sizes for each layer
         assert sizes[0] > 0 and sizes[0] < 1
         assert sizes[1] > 0 and sizes[1] < 1 and sizes[1] > sizes[0]
         tmp = np.linspace(sizes[0], sizes[1], num=(len(from_layers)-1))
         min_sizes = [start_offset] + tmp.tolist()
         max_sizes = tmp.tolist() + [tmp[-1]+start_offset]
         sizes = zip(min_sizes, max_sizes)
    assert len(sizes) == len(from_layers), "sizes and from_layers must have same length"

    loc_pred_layers = []
    cls_pred_layers = []
    pred_layers = []
    anchor_layers = []
    num_classes += 1 # always use background as label 0

    for k, from_layer in enumerate(from_layers):
        from_name = from_layer.name

        # estimate number of anchors per location
        # here I follow the original version in caffe
        # TODO: better way to shape the anchors??
        size = sizes[k]
        assert len(size) > 0, "must provide at least one size"
        size_str = "(" + ",".join([str(x) for x in size]) + ")"
        ratio = ratios[k]
        assert len(ratio) > 0, "must provide at least one ratio"
        ratio_str = "(" + ",".join([str(x) for x in ratio]) + ")"
        num_anchors = len(size) -1 + len(ratio)

        num_loc_pred = num_anchors * 4
        num_cls_pred = num_anchors * num_classes

        pred_conv = bn_relu_conv(from_layer, num_filter=num_loc_pred+num_cls_pred, 
                prefix_name='{}_pred/'.format(from_name), 
                kernel=(3,3), pad=(1,1), no_bias=False, 
                use_global_stats=use_global_stats, fix_gamma=True) # (n ac h w)
        pred_conv = mx.sym.transpose(pred_conv, axes=(0, 2, 3, 1)) # (n h w ac), a=num_anchors
        pred_conv = mx.sym.reshape(pred_conv, (0, -3, -4, num_anchors, -1)) # (n h*w a c)
        pred_conv = mx.sym.reshape(pred_conv, (0, -3, -1)) # (n h*w*a c)
        pred_layers.append(pred_conv)

        # create anchor generation layer
        anchors = mx.sym.MultiBoxPrior(from_layer, sizes=size_str, ratios=ratio_str, \
            clip=clip, name="{}_anchors".format(from_name))
        anchors = mx.sym.reshape(anchors, (-1, 5))
        anchor_layers.append(anchors)

    preds = mx.sym.concat(*pred_layers, num_args=len(anchor_layers), dim=1)
    anchors = mx.sym.concat(*anchor_layers, num_args=len(anchor_layers), dim=1)
    return [preds, anchors]

def get_symbol_train(num_classes, **kwargs):
    '''
    '''
    out_layers = get_hjnet_preact(is_test=False, fix_gamma=False)
    label = mx.sym.Varaible(name='label')

    from_layers = []
    # build hyperfeatures
    hyper_names = ['hyper24', 'hyper48', 'hyper96']
    scales = [8, 4, 2]
    for i in range(3):
        hyper_layer = build_hyperfeature(out_layers[i], out_layers[3], hyper_names[i], 
                num_filter_proj=64, num_filter_hyper=128, scale=scales[i], use_global_stats=False)
        from_layers.append(hyper_layer)

    # 192
    proj192 = bn_relu_conv(out_layers[3], prefix_name='hyper192/proj/', num_filter=64, 
            kernel=(3,3), pad=(1,1), no_bias=True, 
            use_global_stats=False, fix_gamma=True)
    conv192 = bn_relu_conv(data=proj192, prefix_name='hyper192/conv/', num_filter=128, 
            kernel=(1,1), no_bias=True, 
            use_global_stats=False, fix_gamma=True)
    from_layers.append(conv192)

    sizes = []
    for i in (8, 4, 2, 1):
        sizes.append([1.0 / i, np.sqrt(2.0) / i / 2.0])
    ratios = [1.0, 0.8, 1.25]
    clip = True

    preds, anchors = multibox_layer(from_layers, num_classes, 
            sizes=sizes, ratios=ratios, 
            use_global_stats=use_global_stats, clip=clip)

    tmp = mx.symbol.Custom(*[preds, anchors, label], name='anchor_target', op_type='anchor_target')
    pred_target = tmp[0]
    target_cls = tmp[1]
    target_reg = tmp[2]
    mask_reg = tmp[3]

    pred_cls = mx.sym.slice_axis(pred_target, axis=1, begin=0, end=num_classes+1)
    pred_reg = mx.sym.slice_axis(pred_target, axis=1, begin=num_classes+1, end=None)

    cls_loss = mx.symbol.SoftmaxOutput(data=pred_cls, label=target_cls, \
        ignore_label=-1, use_ignore=True, grad_scale=1., multi_output=True, \
        normalization='batch', name="cls_prob")
    # cls_prob = mx.symbol.SoftmaxOutput(data=cls_preds, label=cls_target, \
    #     ignore_label=-1, use_ignore=True, grad_scale=3., multi_output=True, \
    #     normalization='valid', name="cls_prob")
    loc_diff = pred_reg - target_reg
    masked_loc_diff = mx.sym.broadcast_mull(loc_diff, mask_reg)
    loc_loss_ = mx.symbol.smooth_l1(name="loc_loss_", data=masked_loc_diff, scalar=1.0)
    loc_loss = mx.symbol.MakeLoss(loc_loss_, grad_scale=0.1, \
        normalization='batch', name="loc_loss")

    # group output
    out = mx.symbol.Group([cls_loss, loc_loss])
    return out
