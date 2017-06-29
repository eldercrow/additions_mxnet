import mxnet as mx
from pva101_multibox import pvanet_multibox
# from symbol.label_mapping_layer import *
# from symbol.reweight_loss_layer import *
from symbol.multibox_target import *
from symbol.softmax_loss import *
from symbol.anchor_target_layer import *
from symbol.multibox_detection_layer import *


def get_symbol_train(num_classes=20, nms_thresh=0.5, force_suppress=False, nms_topk=400):
    '''
    '''
    data = mx.symbol.Variable(name="data")
    label = mx.symbol.Variable(name="label")

    use_global_stats = False
    no_bias = False

    preds, anchors = pvanet_multibox(data, num_classes, 512, use_global_stats, no_bias)
    cls_preds = mx.sym.slice_axis(preds, axis=2, begin=0, end=num_classes)
    loc_preds = mx.sym.slice_axis(preds, axis=2, begin=num_classes, end=None)

    tmp = mx.symbol.Custom(*[cls_preds, loc_preds, anchors, label], op_type='multibox_target', 
            name='multibox_target', n_class=2, variances=(0.1, 0.1, 0.2, 0.2))
    sample_cls = tmp[0]
    sample_reg = tmp[1]
    target_cls = tmp[2]
    target_reg = tmp[3]
    mask_reg = tmp[4]

    # classification
    cls_prob = mx.symbol.SoftmaxOutput(data=sample_cls, label=target_cls, \
        ignore_label=-1, use_ignore=True, grad_scale=1.0, 
        normalization='null', name="cls_prob", out_grad=True)
    cls_loss = mx.symbol.Custom(cls_prob, target_cls, op_type='softmax_loss', 
            ignore_label=-1, use_ignore=True)
    # alpha_cls = mx.sym.var(name='cls_beta', shape=(1,), lr_mult=0.1, wd_mult=0.0)
    # cls_loss_w = cls_loss * mx.sym.exp(-alpha_cls) + 10.0 * alpha_cls
    cls_loss = mx.sym.MakeLoss(cls_loss, name='cls_loss')

    # regression
    loc_diff = sample_reg - target_reg
    masked_loc_diff = mx.sym.broadcast_mul(loc_diff, mask_reg)
    loc_loss = mx.symbol.smooth_l1(name="loc_loss_", data=masked_loc_diff, scalar=1.0)
    loc_loss = mx.sym.sum(loc_loss) * 0.165
    # alpha_loc = mx.sym.var(name='loc_beta', shape=(1,), 
    #         lr_mult=0.1, wd_mult=0.0, init=mx.init.Constant(2.0))
    # loc_loss_w = loc_loss * mx.sym.exp(-alpha_loc) + 10.0 * alpha_loc
    loc_loss = mx.symbol.MakeLoss(loc_loss, grad_scale=1.0, \
        normalization='null', name="loc_loss")

    label_cls = mx.sym.BlockGrad(target_cls, name='label_cls')
    label_reg = mx.sym.BlockGrad(target_reg, name='label_reg')

    # group output
    out = mx.symbol.Group([cls_loss, loc_loss, label_cls, label_reg, mx.sym.BlockGrad(cls_prob)])
    # out = mx.symbol.Group([cls_loss_w, loc_loss_w, label_cls, label_reg, mx.sym.BlockGrad(cls_loss)])
    return out


def get_symbol(num_classes=21, nms_thresh=0.5, force_suppress=False, nms_topk=400):
    '''
    '''
    data = mx.symbol.Variable(name="data")
    use_global_stats = True
    no_bias = False
    th_pos = 0.25
    th_nms = nms_thresh

    preds, anchors = pvanet_multibox(data, num_classes, 512, use_global_stats, no_bias)
    preds_cls = mx.sym.slice_axis(preds, axis=2, begin=0, end=num_classes)
    preds_reg = mx.sym.slice_axis(preds, axis=2, begin=num_classes, end=None)

    probs_cls = mx.sym.reshape(preds_cls, shape=(-1, num_classes))
    probs_cls = mx.sym.SoftmaxActivation(probs_cls)
    probs_cls = mx.sym.slice_axis(probs_cls, axis=1, begin=1, end=None)

    tmp = mx.symbol.Custom(*[probs_cls, preds_reg, anchors], op_type='multibox_detection', 
            name='multibox_detection', th_pos=th_pos, n_class=num_classes-1, th_nms=th_nms, max_detection=300)
    return tmp
