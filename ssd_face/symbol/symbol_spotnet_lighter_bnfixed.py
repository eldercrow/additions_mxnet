import numpy as np
import mxnet as mx
from spotnet_lighter import get_spotnet
from layer.multibox_target import MultiBoxTarget, MultiBoxTargetProp
from layer.multibox_detection import MultiBoxDetection, MultiBoxDetectionProp
from layer.softmax_loss import SoftmaxLoss, SoftmaxLossProp
# from multibox_target import *
# from anchor_target_layer import *
# from multibox_detection import *

def get_symbol_train(num_classes, **kwargs):
    '''
    '''
    fix_bn = True
    n_group = 7
    patch_size = 768
    if 'n_group' in kwargs:
        n_group = kwargs['n_group']
    if 'patch_size' in kwargs:
        patch_size = kwargs['patch_size']

    preds, anchors = get_spotnet(num_classes, patch_size, 
            use_global_stats=fix_bn, n_group=n_group)
    preds_cls = mx.sym.slice_axis(preds, axis=2, begin=0, end=num_classes)
    preds_reg = mx.sym.slice_axis(preds, axis=2, begin=num_classes, end=None)

    label = mx.sym.var(name='label')

    tmp = mx.symbol.Custom(*[preds_cls, preds_reg, anchors, label], op_type='multibox_target', 
            name='multibox_target', n_class=2, variances=(0.1, 0.1, 0.2, 0.2))
    sample_cls = tmp[0]
    sample_reg = tmp[1]
    target_cls = tmp[2]
    target_reg = tmp[3]
    mask_reg = tmp[4]

    cls_loss = mx.symbol.SoftmaxOutput(data=sample_cls, label=target_cls, \
        ignore_label=-1, use_ignore=True, grad_scale=1.0, 
        normalization='null', name="cls_prob", out_grad=True)
    cls_loss = mx.symbol.Custom(cls_loss, target_cls, op_type='softmax_loss', 
            ignore_label=-1, use_ignore=True)
    alpha_cls = mx.sym.var(name='cls_beta', shape=(1,), lr_mult=1.0, wd_mult=0.0)
    cls_loss = cls_loss * mx.sym.exp(-alpha_cls) + alpha_cls * 10.0
    cls_loss = mx.sym.MakeLoss(cls_loss, name='cls_loss')
    loc_diff = sample_reg - target_reg
    masked_loc_diff = mx.sym.broadcast_mul(loc_diff, mask_reg)
    loc_loss = mx.symbol.smooth_l1(name="loc_loss_", data=masked_loc_diff, scalar=1.0)
    loc_loss = mx.sym.sum(loc_loss) 
    alpha_loc = mx.sym.var(name='loc_beta', shape=(1,), 
            lr_mult=1.0, wd_mult=0.0, init=mx.init.Constant(2.0))
    loc_loss = loc_loss * mx.sym.exp(-alpha_loc) + alpha_loc * 10.0
    loc_loss = mx.symbol.MakeLoss(loc_loss, grad_scale=1.0, \
        normalization='null', name="loc_loss")
    # cls_loss = mx.symbol.SoftmaxOutput(data=sample_cls, label=target_cls, \
    #     ignore_label=-1, use_ignore=True, grad_scale=1.0, 
    #     normalization='null', name="cls_prob")
    # loc_diff = sample_reg - target_reg
    # masked_loc_diff = mx.sym.broadcast_mul(loc_diff, mask_reg)
    # loc_loss_ = mx.symbol.smooth_l1(name="loc_loss_", data=masked_loc_diff, scalar=1.0)
    # loc_loss = mx.symbol.MakeLoss(loc_loss_, grad_scale=0.1, \
    #     normalization='null', name="loc_loss")

    label_cls = mx.sym.BlockGrad(target_cls, name='label_cls')
    label_reg = mx.sym.BlockGrad(target_reg, name='label_reg')

    # group output
    out = mx.symbol.Group([cls_loss, loc_loss, label_cls, label_reg])
    return out

def get_symbol(num_classes, **kwargs):
    '''
    '''
    # im_scale = mx.sym.Variable(name='im_scale')

    fix_bn = True
    n_group = 7
    patch_size = 768
    th_pos = 0.25
    th_nms = 1.0 / 3.0
    if 'n_group' in kwargs:
        n_group = kwargs['n_group']
    if 'patch_size' in kwargs:
        patch_size = kwargs['patch_size']
    if 'th_pos' in kwargs:
        th_pos = kwargs['th_pos']
    if 'nms' in kwargs:
        th_nms = kwargs['nms']

    preds, anchors = get_spotnet(num_classes, patch_size, 
            use_global_stats=fix_bn, n_group=n_group)
    preds_cls = mx.sym.slice_axis(preds, axis=2, begin=0, end=num_classes)
    preds_reg = mx.sym.slice_axis(preds, axis=2, begin=num_classes, end=None)

    probs_cls = mx.sym.reshape(preds_cls, shape=(-1, num_classes))
    probs_cls = mx.sym.SoftmaxActivation(probs_cls)
    probs_cls = mx.sym.slice_axis(probs_cls, axis=1, begin=1, end=None)

    tmp = mx.symbol.Custom(*[probs_cls, preds_reg, anchors], op_type='multibox_detection', 
            name='multibox_detection', th_pos=th_pos, n_class=num_classes-1, th_nms=th_nms, max_detection=1000)
    return tmp

if __name__ == '__main__':
    import os
    os.environ['MXNET_ENGINE_TYPE'] = 'NaiveEngine'
    net = get_symbol_train(2, n_group=7, patch_size=768)

    mod = mx.mod.Module(net, data_names=['data'], label_names=['label'])
    mod.bind(data_shapes=[('data', (2, 3, 768, 768))], label_shapes=[('label', (2, 5))])
    mod.init_params()
    args, auxs = mod.get_params()
    for k, v in sorted(args.items()):
        print k + ': ' + str(v.shape)
    for k, v in sorted(auxs.items()):
        print k + ': ' + str(v.shape)
