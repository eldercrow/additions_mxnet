import mxnet as mx
from spotnet_ron import get_spotnet
from layer.anchor_target_ron_layer import *


def get_symbol_train(num_classes, **kwargs):
    '''
    '''
    fix_bn = False
    patch_size = 256
    if 'patch_size' in kwargs:
        patch_size = kwargs['patch_size']
    per_cls_reg = False

    preds, anchors, rpns = get_spotnet(num_classes, use_global_stats=fix_bn, patch_size=patch_size)
    label = mx.sym.var(name='label')

    tmp = mx.symbol.Custom(*[preds, rpns, anchors, label], name='anchor_target', op_type='anchor_target_ron',
            n_class=num_classes, per_cls_reg=per_cls_reg)
    sample_pred = tmp[0]
    target_cls = tmp[1]
    target_reg = tmp[2]
    mask_reg = tmp[3]
    sample_rpn = tmp[4]
    target_rpn = tmp[5]

    pred_cls = mx.sym.slice_axis(sample_pred, axis=1, begin=0, end=num_classes)
    pred_reg = mx.sym.slice_axis(sample_pred, axis=1, begin=num_classes, end=None)

    # classification
    cls_loss = mx.symbol.SoftmaxOutput(data=pred_cls, label=target_cls, \
        ignore_label=-1, use_ignore=True, grad_scale=1.0, 
        normalization='null', name="cls_loss", out_grad=False)

    # regression
    loc_diff = mask_reg * (pred_reg - target_reg)
    loc_loss = mx.symbol.smooth_l1(name="loc_loss_", data=loc_diff, scalar=1.0)
    loc_loss = mx.symbol.MakeLoss(loc_loss, grad_scale=0.2, \
        normalization='null', name="loc_loss")

    rpn_loss = mx.symbol.SoftmaxOutput(data=sample_rpn, label=target_rpn, \
        ignore_label=-1, use_ignore=True, grad_scale=1.0, 
        normalization='null', name="rpn_loss", out_grad=False)

    label_cls = mx.sym.BlockGrad(target_cls, name='label_cls')
    label_reg = mx.sym.BlockGrad(mask_reg, name='label_reg')
    label_rpn = mx.sym.BlockGrad(target_rpn, name='label_rpn')

    # group output
    out = mx.symbol.Group([cls_loss, loc_loss, rpn_loss, label_cls, label_reg, label_rpn])
    return out
