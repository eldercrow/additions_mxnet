from pvtnet_preact import get_pvtnet_preact
from net_block_clone import bn_relu_conv, clone_bn_relu_conv
from anchor_target_layer import *
from multibox_prior_layer import *
import numpy as np

def build_hyperfeature(data, ctx_data, name, num_filter_proj, num_filter_hyper, scale, use_global_stats):
    """
    """
    ctx_proj = bn_relu_conv(data=ctx_data, prefix_name=name+'/proj/', 
            num_filter=num_filter_proj, kernel=(3,3), pad=(1,1), 
            use_global_stats=use_global_stats, fix_gamma=False)
    ctx_up = mx.symbol.UpSampling(ctx_proj, num_args=1, name=name+'/up', scale=scale, sample_type='nearest')
    concat = mx.symbol.Concat(data, ctx_up)
    hyper = bn_relu_conv(data=concat, prefix_name=name+'/conv/', 
            num_filter=num_filter_hyper, kernel=(1,1), 
            use_global_stats=use_global_stats, fix_gamma=False)
    return hyper

def multibox_layer(from_layers, num_classes, sizes, ratios, use_global_stats, clip=True, clone_idx=[]):
    ''' multibox layer '''
    # parameter check
    assert len(from_layers) > 0, "from_layers must not be empty list"
    assert num_classes > 1, "num_classes {} must be larger than 1".format(num_classes)
    assert len(ratios) > 0, "aspect ratios must not be empty list"
    # if not isinstance(ratios[0], list):
    #     # provided only one ratio list, broadcast to all from_layers
    #     ratios = [ratios] * len(from_layers)
    # assert len(ratios) == len(from_layers), "ratios and from_layers must have same length"
    # assert len(sizes) > 0, "sizes must not be empty list"
    # if len(sizes) == 2 and not isinstance(sizes[0], list):
    #     # provided size range, we need to compute the sizes for each layer
    #      assert sizes[0] > 0 and sizes[0] < 1
    #      assert sizes[1] > 0 and sizes[1] < 1 and sizes[1] > sizes[0]
    #      tmp = np.linspace(sizes[0], sizes[1], num=(len(from_layers)-1))
    #      min_sizes = [start_offset] + tmp.tolist()
    #      max_sizes = tmp.tolist() + [tmp[-1]+start_offset]
    #      sizes = zip(min_sizes, max_sizes)
    assert len(sizes) == len(from_layers), "sizes and from_layers must have same length"

    loc_pred_layers = []
    cls_pred_layers = []
    pred_layers = []
    anchor_layers = []
    # num_classes += 1 # always use background as label 0
    #
    if len(clone_idx) > 1:
        clone_ref = clone_idx[0]
        clone_idx = clone_idx[1:]
    else:
        clone_ref = -1
        clone_idx = []

    for k, from_layer in enumerate(from_layers):
        from_name = from_layer.name

        # estimate number of anchors per location
        # here I follow the original version in caffe
        # TODO: better way to shape the anchors??
        size = sizes[k]
        assert len(size) > 0, "must provide at least one size"
        size_str = "(" + ",".join([str(x) for x in size]) + ")"
        # assert len(ratio) > 0, "must provide at least one ratio"
        ratio_str = "(" + ",".join([str(x) for x in ratios]) + ")"
        num_anchors = len(sizes[k]) * len(ratios)

        num_loc_pred = num_anchors * 4
        num_cls_pred = num_anchors * num_classes

        if k == clone_ref:
            pred_conv, ref_syms = bn_relu_conv(from_layer, prefix_name='{}_pred/'.format(from_name), 
                    num_filter=num_loc_pred+num_cls_pred, kernel=(3,3), pad=(1,1), no_bias=False, 
                    use_global_stats=use_global_stats, fix_gamma=False, get_syms=True) # (n ac h w)
        elif k in clone_idx:
            pred_conv = clone_bn_relu_conv(from_layer, prefix_name='{}_pred/'.format(from_name), 
                    src_syms=ref_syms)
        else:
            pred_conv = bn_relu_conv(from_layer, prefix_name='{}_pred/'.format(from_name), 
                    num_filter=num_loc_pred+num_cls_pred, kernel=(3,3), pad=(1,1), no_bias=False, 
                    use_global_stats=use_global_stats, fix_gamma=False) # (n ac h w)

        pred_conv = mx.sym.transpose(pred_conv, axes=(0, 2, 3, 1)) # (n h w ac), a=num_anchors
        pred_conv = mx.sym.reshape(pred_conv, shape=(0, -3, -4, num_anchors, -1)) # (n h*w a c)
        pred_conv = mx.sym.reshape(pred_conv, shape=(0, -3, -1)) # (n h*w*a c)
        pred_layers.append(pred_conv)

        # create anchor generation layer
        # if k == 0:
        #     anchors, anchor_scales = mx.sym.Custom(from_layer, op_type='multibox_prior_python', 
        #             sizes=sizes[k], ratios=ratios)
        # else:
        #     anchors = mx.sym.Pooling(anchors, kernel=(2,2), stride=(2,2), pool_type='avg',
        #             attr={'__lr_mult__': 0.0})
        #     anchors = mx.sym.broadcast_mul(anchors, anchor_scales)
        # anchorsf = mx.sym.transpose(anchors, axes=(0, 2, 3, 1))
        # anchorsf = mx.sym.reshape(anchorsf, shape=(-1, 4))
        # anchor_layers.append(anchorsf)
        anchors = mx.contrib.symbol.MultiBoxPrior(from_layer, sizes=size_str, ratios=ratio_str, \
            clip=clip, name="{}_anchors".format(from_name))
        anchors = mx.sym.reshape(anchors, shape=(-1, 4))
        anchor_layers.append(anchors)

    preds = mx.sym.concat(*pred_layers, num_args=len(anchor_layers), dim=1)
    anchors = mx.sym.concat(*anchor_layers, num_args=len(anchor_layers), dim=0)
    return [preds, anchors]

def get_symbol_train(num_classes, **kwargs):
    '''
    '''
    out_layers, ctx_layer = get_pvtnet_preact(use_global_stats=False, fix_gamma=False)
    label = mx.sym.var(name='label')

    from_layers = []
    # build hyperfeatures
    hyper_names = ['hyper12', 'hyper24', 'hyper48', 'hyper96']
    scales = [16, 8, 4, 2]
    for i, s in enumerate(scales):
        hyper_layer = build_hyperfeature(out_layers[i], ctx_layer, name=hyper_names[i], 
                num_filter_proj=s*6, num_filter_hyper=128, scale=s, use_global_stats=False)
        from_layers.append(hyper_layer)

    # 192
    conv192, src_syms = bn_relu_conv(out_layers[4], prefix_name='hyper192/conv/', 
            num_filter=128, kernel=(1,1), 
            use_global_stats=False, fix_gamma=False, get_syms=True)
    from_layers.append(conv192)

    # remaining clone layers
    clone_idx = []
    # clone_idx = [4]
    # for i in range(5, len(out_layers)):
    #     rf = (2**i) * 12
    #     prefix_name = 'hyper{}/conv'.format(rf)
    #     conv_ = clone_bn_relu_conv(out_layers[i], prefix_name=prefix_name, src_syms=src_syms)
    #     from_layers.append(conv_)
    #     clone_idx.append[i]

    rfs = [12.0, 24.0, 48.0, 96.0, 192.0]
    n_from_layers = len(from_layers)
    sizes = []
    for i in range(n_from_layers):
        s = rfs[i] / 256.0
        sizes.append([s, s / np.sqrt(2.0)])
    ratios = [1.0, 0.8, 1.25]
    clip = True

    preds, anchors = multibox_layer(from_layers, num_classes, 
            sizes=sizes, ratios=ratios, 
            use_global_stats=False, clip=clip, clone_idx=clone_idx)

    tmp = mx.symbol.Custom(*[preds, anchors, label], name='anchor_target', op_type='anchor_target')
    pred_target = tmp[0]
    target_cls = tmp[1]
    target_reg = tmp[2]
    mask_reg = tmp[3]

    pred_cls = mx.sym.slice_axis(pred_target, axis=1, begin=0, end=num_classes)
    pred_reg = mx.sym.slice_axis(pred_target, axis=1, begin=num_classes, end=None)

    cls_loss = mx.symbol.SoftmaxOutput(data=pred_cls, label=target_cls, \
        ignore_label=-1, use_ignore=True, grad_scale=3.0, #multi_output=True, \
        normalization='valid', name="cls_prob")
    # cls_prob = mx.symbol.SoftmaxOutput(data=cls_preds, label=cls_target, \
    #     ignore_label=-1, use_ignore=True, grad_scale=3., multi_output=True, \
    #     normalization='valid', name="cls_prob")
    loc_diff = pred_reg - target_reg
    masked_loc_diff = mx.sym.broadcast_mul(loc_diff, mask_reg)
    loc_loss_ = mx.symbol.smooth_l1(name="loc_loss_", data=masked_loc_diff, scalar=1.0)
    loc_loss = mx.symbol.MakeLoss(loc_loss_, grad_scale=1.0, \
        normalization='valid', name="loc_loss")

    label_cls = mx.sym.MakeLoss(target_cls, grad_scale=0, name='label_cls')
    label_reg = mx.sym.MakeLoss(target_reg, grad_scale=0, name='label_reg')

    # group output
    out = mx.symbol.Group([cls_loss, loc_loss, label_cls, label_reg])
    return out

if __name__ == '__main__':
    import os
    os.environ['MXNET_ENGINE_TYPE'] = 'NaiveEngine'
    net = get_symbol_train(2)

    mod = mx.mod.Module(net, data_names=['data'], label_names=['label'])
    mod.bind(data_shapes=[('data', (8, 3, 192, 192))], label_shapes=[('label', (8, 5))])
    mod.init_params()

