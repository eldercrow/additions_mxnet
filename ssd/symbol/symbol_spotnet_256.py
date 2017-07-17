import mxnet as mx
from spotnet_voc import get_spotnet
# from layer.multibox_target_layer import MultiBoxTarget, MultiBoxTargetProp
# from layer.multibox_detection_layer import MultiBoxDetection, MultiBoxDetectionProp
# from layer.multibox_target2_layer import *
# from layer.softmax_loss_layer import SoftmaxLoss, SoftmaxLossProp
from layer.anchor_target_layer import *


def get_symbol_train(num_classes, **kwargs):
    '''
    '''
    fix_bn = False
    patch_size = 256
    if 'patch_size' in kwargs:
        patch_size = kwargs['patch_size']
    per_cls_reg = False

    preds, anchors = get_spotnet(num_classes, patch_size, per_cls_reg, use_global_stats=fix_bn)
    label = mx.sym.var(name='label')

    tmp = mx.symbol.Custom(*[preds, anchors, label], name='anchor_target', op_type='anchor_target',
            n_class=num_classes, per_cls_reg=per_cls_reg)
    pred_target = tmp[0]
    target_cls = tmp[1]
    target_reg = tmp[2]
    mask_reg = tmp[3]

    pred_cls = mx.sym.slice_axis(pred_target, axis=1, begin=0, end=num_classes)
    pred_reg = mx.sym.slice_axis(pred_target, axis=1, begin=num_classes, end=None)

    # classification
    cls_loss = mx.symbol.SoftmaxOutput(data=pred_cls, label=target_cls, \
        ignore_label=-1, use_ignore=True, grad_scale=1.0, 
        normalization='null', name="cls_loss", out_grad=False)

    # regression
    loc_diff = mask_reg * (pred_reg - target_reg)
    loc_loss = mx.symbol.smooth_l1(name="loc_loss_", data=loc_diff, scalar=1.0)
    loc_loss = mx.symbol.MakeLoss(loc_loss, grad_scale=0.2, \
        normalization='null', name="loc_loss")

    label_cls = mx.sym.BlockGrad(target_cls, name='label_cls')
    label_reg = mx.sym.BlockGrad(mask_reg, name='label_reg')

    # group output
    out = mx.symbol.Group([cls_loss, loc_loss, label_cls, label_reg])
    return out
