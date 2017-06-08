from spotnet_lighter import get_spotnet
from layer.anchor_target_layer import *
from layer.softmax_loss import SoftmaxLoss, SoftmaxLossProp
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
        normalization='null', name="cls_prob")
    cls_loss = mx.symbol.Custom(cls_loss, target_cls, op_type='softmax_loss', 
            ignore_label=-1, use_ignore=True)
    alpha_cls = mx.sym.var(name='cls_beta', shape=(1,))
    cls_loss = cls_loss * mx.sym.exp(-alpha_cls) + alpha_cls
    cls_loss = mx.sym.MakeLoss(cls_loss, name='cls_loss')
    loc_diff = pred_reg - target_reg
    masked_loc_diff = mx.sym.broadcast_mul(loc_diff, mask_reg)
    loc_loss = mx.symbol.smooth_l1(name="loc_loss_", data=masked_loc_diff, scalar=1.0)
    loc_loss = mx.sym.sum(loc_loss) 
    alpha_loc = mx.sym.var(name='loc_beta', shape=(1,))
    loc_loss = loc_loss * mx.sym.exp(-alpha_loc) + alpha_loc
    loc_loss = mx.symbol.MakeLoss(loc_loss, grad_scale=1.0, \
        normalization='null', name="loc_loss")
    # loc_diff = pred_reg - target_reg
    # masked_loc_diff = mx.sym.broadcast_mul(loc_diff, mask_reg)
    # loc_loss_ = mx.symbol.smooth_l1(name="loc_loss_", data=masked_loc_diff, scalar=1.0)
    # loc_loss = mx.symbol.MakeLoss(loc_loss_, grad_scale=1.0, \
    #     normalization='valid', name="loc_loss")

    label_cls = mx.sym.BlockGrad(target_cls, name='label_cls')
    label_reg = mx.sym.BlockGrad(target_reg, name='label_reg')

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

    args, auxs = mod.get_params()
    for k, v in sorted(args.items()):
        print k + ': ' + str(v.shape)
    for k, v in sorted(auxs.items()):
        print k + ': ' + str(v.shape)

