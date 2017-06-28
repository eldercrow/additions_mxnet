import mxnet as mx
from pva100_multibox import pvanet_multibox
from symbol.label_mapping_layer import *
from symbol.reweight_loss_layer import *
from symbol.anchor_target_layer import *


def get_symbol_train(num_classes=20, nms_thresh=0.5, force_suppress=False, nms_topk=400):
    '''
    '''
    data = mx.symbol.Variable(name="data")
    label = mx.symbol.Variable(name="label")

    use_global_stats = False
    no_bias = False

    preds, anchor_boxes = pvanet_multibox(data, num_classes, 512, use_global_stats, no_bias)
    cls_preds = mx.sym.slice_axis(preds, axis=2, begin=0, end=num_classes)
    cls_preds = mx.sym.transpose(cls_preds, axes=(0, 2, 1))
    loc_preds = mx.sym.slice_axis(preds, axis=2, begin=num_classes, end=None)

    cls_target, loc_target, loc_target_mask = \
            mx.sym.Custom(anchor_boxes, label, name='label_mapping', op_type='label_mapping')

    cls_prob = mx.symbol.SoftmaxOutput(data=cls_preds, label=cls_target, \
        ignore_label=-1, use_ignore=True, grad_scale=1., multi_output=True, \
        normalization='null', name="cls_prob", out_grad=True)
    cls_prob, cls_target_ohem = mx.sym.Custom(cls_prob, cls_target, op_type='reweight_loss', rand_mult=6)
    cls_loss = mx.sym.MakeLoss(cls_prob, name='cls_prob_loss', grad_scale=1.)

    loc_diff = mx.symbol.smooth_l1(name="loc_diff", \
        data=loc_target_mask * (loc_preds - loc_target), scalar=1.0)
    loc_loss = mx.symbol.MakeLoss(loc_diff, grad_scale=0.2, \
        normalization='null', name="loc_loss")

    # monitoring training status
    cls_label = mx.symbol.BlockGrad(data=cls_target_ohem, name="cls_label")
    loc_label = mx.symbol.BlockGrad(data=loc_target_mask, name='loc_label')
    # det = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
    #     name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress,
    #     variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)
    # det = mx.symbol.BlockGrad(data=det, name="det_out")

    # group output
    out = mx.symbol.Group([cls_loss, loc_loss, cls_label, loc_label])
    # out = mx.symbol.Group([cls_prob, loc_loss, cls_label, det])
    return out


def get_symbol(num_classes=20, nms_thresh=0.5, force_suppress=False, nms_topk=400):
    '''
    '''
    data = mx.symbol.Variable(name="data")
    use_global_stats = True
    no_bias = False

    loc_preds, cls_preds, anchor_boxes = pvanet_multibox(data, num_classes, use_global_stats, no_bias)

    cls_prob = mx.symbol.SoftmaxActivation(data=cls_preds, mode='channel', \
        name='cls_prob')
    out = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
        name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress,
        variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)
    return out
