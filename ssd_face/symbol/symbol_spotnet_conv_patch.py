from spotnet_conv import get_spotnet
from anchor_target_layer import *
import numpy as np

def get_symbol_train(num_classes, **kwargs):
    '''
    '''
    fix_bn = False
    n_group = 5
    patch_size = 256
    if 'n_group' in kwargs:
        n_group = kwargs['n_group']
    if 'patch_size' in kwargs:
        patch_size = kwargs['patch_size']

    preds, anchors = get_spotnet(num_classes, patch_size, 
            use_global_stats=fix_bn, n_group=n_group)
    label = mx.sym.var(name='label')

    tmp = mx.symbol.Custom(*[preds, anchors, label], name='anchor_target', op_type='anchor_target')
    pred_target = tmp[0]
    target_cls = tmp[1]
    target_reg = tmp[2]
    mask_reg = tmp[3]

    pred_cls = mx.sym.slice_axis(pred_target, axis=1, begin=0, end=num_classes)
    pred_reg = mx.sym.slice_axis(pred_target, axis=1, begin=num_classes, end=None)

    cls_loss = mx.symbol.SoftmaxOutput(data=pred_cls, label=target_cls, \
        ignore_label=-1, use_ignore=True, grad_scale=3.0, 
        normalization='valid', name="cls_prob")
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
    net = get_symbol_train(2, n_group=5, patch_size=256)

    mod = mx.mod.Module(net, data_names=['data'], label_names=['label'])
    mod.bind(data_shapes=[('data', (2, 3, 256, 256))], label_shapes=[('label', (2, 5))])
    mod.init_params()

    import ipdb
    ipdb.set_trace()

    args, auxs = mod.get_params()
    for k, v in sorted(args.items()):
        print k + ': ' + str(v.shape)
    for k, v in sorted(auxs.items()):
        print k + ': ' + str(v.shape)


