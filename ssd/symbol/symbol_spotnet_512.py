import mxnet as mx
import numpy as np
from spotnet_multibox import get_spotnet
from layer.label_mapping_layer import *
from layer.reweight_loss_layer import *
# from layer.multibox_target2_layer import MultiBoxTarget2, MultiBoxTargetProp2
# from layer.multibox_target_layer import MultiBoxTarget, MultiBoxTargetProp
# from layer.softmax_loss_layer import SoftmaxLoss, SoftmaxLossProp
from layer.multibox_detection_layer import MultiBoxDetection, MultiBoxDetectionProp
from layer.anchor_target_layer import *

def get_symbol_train(num_classes, **kwargs):
    '''
    '''
    fix_bn = False
    patch_size = 512
    per_cls_reg = False
    if 'patch_size' in kwargs:
        patch_size = kwargs['patch_size']

    preds, anchors = get_spotnet(num_classes, patch_size, per_cls_reg, use_global_stats=fix_bn)
    preds_cls = mx.sym.slice_axis(preds, axis=2, begin=0, end=num_classes) # (n_batch, n_anc, n_cls)
    preds_cls = mx.sym.transpose(preds_cls, (0, 2, 1))
    probs_cls = mx.sym.SoftmaxActivation(preds_cls, mode='channel')
    preds_reg = mx.sym.slice_axis(preds, axis=2, begin=num_classes, end=None)

    label = mx.sym.var(name='label')

    cls_target, bbox_target, bbox_weight, max_iou = mx.sym.Custom(anchors, label, op_type='label_mapping', \
            name='label_mapping', th_iou_neg=1.0 / 3.0)

    probs_cls = mx.sym.SoftmaxOutput(preds_cls, label=cls_target, name='cls_prob',
            ignore_label=-1, use_ignore=True, multi_output=True, normalization='null')
    tmp_in = [probs_cls, cls_target, bbox_weight, max_iou]
    probs_cls, cls_target_ohem, bbox_weight = mx.sym.Custom(*tmp_in, op_type='reweight_loss')

    # regression
    loc_diff = (preds_reg - bbox_target) * bbox_weight
    loc_loss = mx.symbol.smooth_l1(name="loc_loss_", data=loc_diff, scalar=1.0) * 0.2
    loc_loss = mx.symbol.MakeLoss(loc_loss, grad_scale=1.0, normalization='null', name="loc_loss")

    label_cls = mx.sym.BlockGrad(cls_target, name='label_cls')
    label_reg = mx.sym.BlockGrad(bbox_weight, name='label_reg')

    # group output
    out = mx.symbol.Group([probs_cls, loc_loss, label_cls, label_reg])
    # out = mx.symbol.Group([cls_loss_w, loc_loss_w, label_cls, label_reg, mx.sym.BlockGrad(cls_loss)])
    return out

def get_symbol(num_classes, **kwargs):
    '''
    '''
    # im_scale = mx.sym.Variable(name='im_scale')

    fix_bn = True
    patch_size = 512
    per_cls_reg = False
    th_pos = 0.25
    th_nms = 1.0 / 3.0
    per_cls_reg = False
    if 'patch_size' in kwargs:
        patch_size = kwargs['patch_size']
    if 'th_pos' in kwargs:
        th_pos = kwargs['th_pos']
    if 'nms' in kwargs:
        th_nms = kwargs['nms']

    preds, anchors = get_spotnet(num_classes, patch_size, per_cls_reg=per_cls_reg, use_global_stats=fix_bn)
    preds_cls = mx.sym.slice_axis(preds, axis=2, begin=0, end=num_classes)
    preds_reg = mx.sym.slice_axis(preds, axis=2, begin=num_classes, end=None)

    probs_cls = mx.sym.reshape(preds_cls, shape=(-1, num_classes))
    probs_cls = mx.sym.SoftmaxActivation(probs_cls)
    probs_cls = mx.sym.slice_axis(probs_cls, axis=1, begin=1, end=None)

    tmp = mx.symbol.Custom(*[probs_cls, preds_reg, anchors], op_type='multibox_detection',
            name='multibox_detection', th_pos=th_pos, n_class=num_classes-1, th_nms=th_nms, max_detection=128)
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
