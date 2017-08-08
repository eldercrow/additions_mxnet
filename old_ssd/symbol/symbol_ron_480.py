import mxnet as mx
import numpy as np
from spotnet_ron import get_spotnet
from layer.multibox_target_ron_layer import *
from layer.multibox_detection_layer import MultiBoxDetection, MultiBoxDetectionProp
from layer.anchor_target_ron_layer import *

def get_symbol_train(num_classes, **kwargs):
    '''
    '''
    fix_bn = False
    patch_size = 480
    per_cls_reg = True
    if 'patch_size' in kwargs:
        patch_size = kwargs['patch_size']

    preds, anchors, preds_rpn = get_spotnet( \
            num_classes, use_global_stats=fix_bn, patch_size=patch_size, per_cls_reg=per_cls_reg)
    preds_cls = mx.sym.slice_axis(preds, axis=2, begin=0, end=num_classes) # (n_batch, n_anc, n_cls)
    preds_reg = mx.sym.slice_axis(preds, axis=2, begin=num_classes, end=None)
    probs_rpn = mx.sym.SoftmaxActivation(mx.sym.transpose(preds_rpn, (0, 2, 1)), mode='channel')

    label = mx.sym.var(name='label')

    tmp_in = [preds_cls, preds_reg, preds_rpn, anchors, label, probs_rpn]
    tmp = mx.symbol.Custom(*tmp_in, op_type='multibox_target_ron', name='multibox_target',
            n_class=num_classes, img_wh=(patch_size, patch_size), variances=(0.1, 0.1, 0.2, 0.2),
            per_cls_reg=per_cls_reg, normalization=False)
    sample_cls = tmp[0]
    sample_reg = tmp[1]
    sample_rpn = tmp[2]
    target_cls = tmp[3]
    target_reg = tmp[4]
    mask_reg = tmp[5]
    target_rpn = tmp[6]

    # classification
    cls_loss = mx.symbol.SoftmaxOutput(data=sample_cls, label=target_cls, name='cls_prob', \
            ignore_label=-1, use_ignore=True, grad_scale=1.0, normalization='null')

    # regression
    loc_diff = (sample_reg - target_reg) * mask_reg
    loc_loss = mx.symbol.smooth_l1(name="loc_loss_", data=loc_diff, scalar=1.0)
    loc_loss = mx.symbol.MakeLoss(loc_loss, name='loc_loss', grad_scale=0.2)

    # rpn
    rpn_loss = mx.symbol.SoftmaxOutput(data=sample_rpn, label=target_rpn, name='rpn_prob', \
            ignore_label=-1, use_ignore=True, grad_scale=1.0, normalization='null')

    # for metric
    label_cls = mx.sym.BlockGrad(target_cls, name='label_cls')
    label_reg = mx.sym.BlockGrad(mask_reg, name='label_reg')
    label_rpn = mx.sym.BlockGrad(target_rpn, name='label_rpn')

    # group output
    out = mx.symbol.Group([cls_loss, loc_loss, rpn_loss, label_cls, label_reg, label_rpn])
    return out


def get_symbol(num_classes, **kwargs):
    '''
    '''
    # im_scale = mx.sym.Variable(name='im_scale')

    fix_bn = True
    patch_size = 480
    th_pos = 0.25
    th_nms = 1.0 / 3.0
    per_cls_reg = True
    if 'patch_size' in kwargs:
        patch_size = kwargs['patch_size']
    if 'th_pos' in kwargs:
        th_pos = kwargs['th_pos']
    if 'nms' in kwargs:
        th_nms = kwargs['nms']

    preds, anchors, preds_rpn = get_spotnet( 
            num_classes, patch_size=patch_size, use_global_stats=fix_bn, per_cls_reg=per_cls_reg)
    preds_cls = mx.sym.slice_axis(preds, axis=2, begin=0, end=num_classes)
    preds_reg = mx.sym.slice_axis(preds, axis=2, begin=num_classes, end=None)

    probs_rpn = mx.sym.reshape(preds_rpn, shape=(-1, 2))
    probs_rpn = mx.sym.SoftmaxActivation(probs_rpn)
    probs_rpn = mx.sym.slice_axis(probs_rpn, axis=1, begin=1, end=None)

    probs_cls = mx.sym.reshape(preds_cls, shape=(-1, num_classes))
    probs_cls = mx.sym.SoftmaxActivation(probs_cls)
    probs_cls = mx.sym.slice_axis(probs_cls, axis=1, begin=1, end=None)

    probs_cls = mx.sym.broadcast_mul(probs_cls, probs_rpn)

    tmp = mx.symbol.Custom(*[probs_cls, preds_reg, anchors], op_type='multibox_detection',
            name='multibox_detection', th_pos=th_pos, n_class=num_classes-1, th_nms=th_nms, 
            per_cls_reg=per_cls_reg, max_detection=128)
    return tmp

# if __name__ == '__main__':
#     import os
#     os.environ['MXNET_ENGINE_TYPE'] = 'NaiveEngine'
#     net = get_symbol_train(2, n_group=7, patch_size=768)
#
#     mod = mx.mod.Module(net, data_names=['data'], label_names=['label'])
#     mod.bind(data_shapes=[('data', (2, 3, 768, 768))], label_shapes=[('label', (2, 5))])
#     mod.init_params()
#     args, auxs = mod.get_params()
#     for k, v in sorted(args.items()):
#         print k + ': ' + str(v.shape)
#     for k, v in sorted(auxs.items()):
#         print k + ': ' + str(v.shape)
