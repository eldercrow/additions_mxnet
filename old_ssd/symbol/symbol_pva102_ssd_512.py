import mxnet as mx
from pva102_multibox import pvanet_multibox
# from symbol.label_mapping_layer import *
# from symbol.reweight_loss_layer import *
from layer.multibox_target2_layer import *
# from layer.multibox_target_layer import *
# from layer.softmax_loss_layer import *
# from layer.anchor_target_layer import *
from layer.multibox_detection_layer import *


def get_symbol_train(num_classes=21, nms_thresh=0.5, force_suppress=False, nms_topk=400):
    '''
    '''
    data = mx.symbol.Variable(name="data")
    label = mx.symbol.Variable(name="label")

    use_global_stats = False
    no_bias = False
    per_cls_reg = True

    preds, anchors = pvanet_multibox(data, num_classes, 512, per_cls_reg, use_global_stats, no_bias)
    preds_cls = mx.sym.slice_axis(preds, axis=2, begin=0, end=num_classes) # (n_batch, n_anc, n_cls)
    probs_cls = mx.sym.SoftmaxActivation(mx.sym.transpose(preds_cls, (0, 2, 1)), mode='channel')
    preds_reg = mx.sym.slice_axis(preds, axis=2, begin=num_classes, end=None)

    box_ratios = (1.0, 2.0/3.0, 3.0/2.0, 4.0/9.0, 9.0/4.0)
    tmp_in = [preds_cls, preds_reg, anchors, label, probs_cls]
    tmp = mx.symbol.Custom(*tmp_in, op_type='multibox_target2', name='multibox_target2',
            n_class=num_classes, img_wh=(512, 512), variances=(0.1, 0.1, 0.2, 0.2),
            box_ratios=box_ratios, per_cls_reg=per_cls_reg, normalization=True)
    sample_cls = tmp[0]
    sample_reg = tmp[1]
    target_cls = tmp[2]
    target_reg = tmp[3]
    mask_reg = tmp[4]

    # classification
    cls_loss = mx.symbol.SoftmaxOutput(data=sample_cls, label=target_cls, name='cls_prob', \
            ignore_label=-1, use_ignore=True, grad_scale=1.0, normalization='null')

    # regression
    loc_diff = (sample_reg - target_reg) * mask_reg
    loc_loss = mx.symbol.smooth_l1(name="loc_loss_", data=loc_diff, scalar=1.0)
    loc_loss = mx.symbol.MakeLoss(loc_loss, name='loc_loss', grad_scale=0.2)

    # for metric
    label_cls = mx.sym.BlockGrad(target_cls, name='label_cls')
    label_reg = mx.sym.BlockGrad(mask_reg, name='label_reg')

    # group output
    out = mx.symbol.Group([cls_loss, loc_loss, label_cls, label_reg])
    return out


def get_symbol(num_classes=21, nms_thresh=0.5, force_nms=False, nms_topk=400):
    '''
    '''
    data = mx.symbol.Variable(name="data")
    use_global_stats = True
    no_bias = False
    th_pos = 0.25
    th_nms = nms_thresh
    per_cls_reg = True

    preds, anchors = pvanet_multibox(data, num_classes, 512, use_global_stats, no_bias)
    preds_cls = mx.sym.slice_axis(preds, axis=2, begin=0, end=num_classes)
    preds_reg = mx.sym.slice_axis(preds, axis=2, begin=num_classes, end=None)

    probs_cls = mx.sym.reshape(preds_cls, shape=(-1, num_classes))
    probs_cls = mx.sym.SoftmaxActivation(probs_cls)
    probs_cls = mx.sym.slice_axis(probs_cls, axis=1, begin=1, end=None)

    tmp_in = [probs_cls, preds_reg, anchors]
    tmp = mx.symbol.Custom(*tmp_in, op_type='multibox_detection', name='multibox_detection', 
            th_pos=th_pos, n_class=num_classes-1, th_nms=th_nms, per_cls_reg=per_cls_reg, max_detection=300)
    return tmp
