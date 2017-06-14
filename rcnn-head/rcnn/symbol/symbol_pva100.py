import mxnet as mx
import proposal
import proposal_target
from rcnn.config import config
from rcnn.core import softmax_loss


def convolution(data, name, num_filter, kernel, pad, stride=(1,1), no_bias=False, lr_mult=1.0):
    ''' convolution with lr_mult and wd_mult '''
    w = mx.sym.var(name+'_weight', lr_mult=lr_mult, wd_mult=lr_mult)
    b = None
    if no_bias == False:
        b = mx.sym.var(name+'_bias', lr_mult=lr_mult*2.0, wd_mult=0.0)
    conv = mx.sym.Convolution(data, weight=w, bias=b, name=name, num_filter=num_filter, 
            kernel=kernel, pad=pad, stride=stride, no_bias=no_bias)
    return conv


def fullyconnected(data, name, num_hidden, no_bias=False, lr_mult=1.0):
    ''' convolution with lr_mult and wd_mult '''
    w = mx.sym.var(name+'_weight', lr_mult=lr_mult, wd_mult=lr_mult)
    b = None
    if no_bias == False:
        b = mx.sym.var(name+'_bias', lr_mult=lr_mult*2.0, wd_mult=0.0)
    fc = mx.sym.FullyConnected(data, weight=w, bias=b, name=name, num_hidden=num_hidden, no_bias=no_bias)
    return fc


def batchnorm(data, name, use_global_stats, fix_gamma=False, lr_mult=1.0):
    ''' batch norm with lr_mult and wd_mult '''
    g = mx.sym.var(name+'_gamma', lr_mult=lr_mult, wd_mult=0.0)
    b = mx.sym.var(name+'_beta', lr_mult=lr_mult, wd_mult=0.0)
    bn = mx.sym.BatchNorm(data, gamma=g, beta=b, name=name, 
            use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    return bn


def conv_bn_relu(data, group_name, 
        num_filter, kernel, pad, stride, no_bias, 
        use_global_stats=True, use_crelu=False, 
        get_syms=False, lr_mult=1.0):
    ''' use in mCReLU '''
    conv_name = group_name + ''
    concat_name = group_name + '/concat'
    bn_name = group_name + '/bn'
    relu_name = group_name + '/relu'

    syms = {}
    conv = convolution(data, name=conv_name, 
            num_filter=num_filter, kernel=kernel, pad=pad, stride=stride, no_bias=no_bias, lr_mult=lr_mult)
    syms['conv'] = conv
    if use_crelu:
        conv = mx.sym.concat(conv, -conv, name=concat_name)
        syms['concat'] = conv
    bn = batchnorm(conv, name=bn_name, use_global_stats=use_global_stats, fix_gamma=False, lr_mult=lr_mult)
    syms['bn'] = bn
    relu = mx.sym.Activation(bn, name=relu_name, act_type='relu')
    if get_syms:
        return relu, syms
    else:
        return relu


def mcrelu(data, prefix_group, filters, no_bias, use_global_stats, lr_mult=1.0):
    ''' conv2 and conv3 '''
    group1 = conv_bn_relu(data, group_name=prefix_group+'_1', 
            num_filter=filters[0], kernel=(1,1), pad=(0,0), stride=(1,1), no_bias=no_bias,
            use_global_stats=use_global_stats, lr_mult=lr_mult)
    group2 = conv_bn_relu(group1, group_name=prefix_group+'_2',
            num_filter=filters[1], kernel=(3,3), pad=(1,1), stride=(2,2), no_bias=no_bias,
            use_global_stats=use_global_stats, use_crelu=True, lr_mult=lr_mult)
    conv3 = convolution(group2, name=prefix_group+'_3/out',
            num_filter=filters[2], kernel=(1,1), pad=(0,0), stride=(1,1), no_bias=no_bias, lr_mult=lr_mult)
    proj1 = convolution(data, name=prefix_group+'_1/proj',
            num_filter=filters[2], kernel=(1,1), pad=(0,0), stride=(2,2), no_bias=no_bias, lr_mult=lr_mult)
    bn3 = batchnorm(conv3+proj1, name=prefix_group+'_3/elt/bn', 
            use_global_stats=use_global_stats, fix_gamma=False, lr_mult=lr_mult)
    relu3 = mx.sym.Activation(bn3, name=prefix_group+'_3/relu', act_type='relu')
    return relu3


def inception(data, prefix_group, 
        filters_1, filters_3, filters_5, no_bias, 
        use_global_stats, do_pool=False, lr_mult=1.0):
    ''' inception group '''
    if do_pool:
        pool1 = mx.sym.Pooling(data, name=prefix_group+'/pool1', kernel=(3,3), pad=(0,0), stride=(2,2), 
                pool_type='max', pooling_convention='full')
        ss = (2, 2)
    else:
        pool1 = data
        ss = (1, 1)
    # conv1
    conv1 = conv_bn_relu(pool1, group_name=prefix_group+'/conv1', 
            num_filter=filters_1, kernel=(1,1), pad=(0,0), stride=(1,1), no_bias=no_bias,
            use_global_stats=use_global_stats, lr_mult=lr_mult)
    # conv3
    conv3_1 = conv_bn_relu(data, group_name=prefix_group+'/conv3_1', 
            num_filter=filters_3[0], kernel=(1,1), pad=(0,0), stride=(1,1), no_bias=no_bias,
            use_global_stats=use_global_stats, lr_mult=lr_mult)
    conv3_2 = conv_bn_relu(conv3_1, group_name=prefix_group+'/conv3_2', 
            num_filter=filters_3[1], kernel=(3,3), pad=(1,1), stride=ss, no_bias=no_bias,
            use_global_stats=use_global_stats, lr_mult=lr_mult)
    # conv5
    conv5_1 = conv_bn_relu(data, group_name=prefix_group+'/conv5_1', 
            num_filter=filters_5[0], kernel=(1,1), pad=(0,0), stride=(1,1), no_bias=no_bias,
            use_global_stats=use_global_stats, lr_mult=lr_mult)
    conv5_2 = conv_bn_relu(conv5_1, group_name=prefix_group+'/conv5_2', 
            num_filter=filters_5[1], kernel=(3,3), pad=(1,1), stride=(1,1), no_bias=no_bias,
            use_global_stats=use_global_stats, lr_mult=lr_mult)
    conv5_3 = conv_bn_relu(conv5_2, group_name=prefix_group+'/conv5_3', 
            num_filter=filters_5[2], kernel=(3,3), pad=(1,1), stride=ss, no_bias=no_bias,
            use_global_stats=use_global_stats, lr_mult=lr_mult)
    return mx.sym.concat(conv1, conv3_2, conv5_3)


def residual_inc(lhs, rhs, prefix_lhs, prefix_rhs, num_filter, stride, no_bias, use_global_stats, lr_mult=1.0):
    ''' residual connection between inception layers '''
    lhs = convolution(lhs, name=prefix_lhs+'/proj',
            num_filter=num_filter, kernel=(1,1), pad=(0,0), stride=stride, no_bias=no_bias, lr_mult=lr_mult)
    rhs = convolution(rhs, name=prefix_rhs+'/out',
            num_filter=num_filter, kernel=(1,1), pad=(0,0), stride=(1,1), no_bias=no_bias, lr_mult=lr_mult)
    elt = lhs+rhs
    bn = batchnorm(elt, name=prefix_rhs+'/elt/bn',
            use_global_stats=use_global_stats, fix_gamma=False, lr_mult=lr_mult)
    relu = mx.sym.Activation(bn, act_type='relu')
    return relu, elt


def pvanet_preact(data, use_global_stats=True, no_bias=False, lr_mult=1.0):
    ''' pvanet 10.0 '''
    conv1 = conv_bn_relu(data, group_name='conv1', 
            num_filter=16, kernel=(4,4), pad=(1,1), stride=(2,2), no_bias=no_bias, 
            use_global_stats=use_global_stats, use_crelu=True, lr_mult=lr_mult)
    # conv2
    conv2 = mcrelu(conv1, prefix_group='conv2', 
            filters=(16, 24, 48), no_bias=no_bias, use_global_stats=use_global_stats, lr_mult=lr_mult)
    # conv3
    conv3 = mcrelu(conv2, prefix_group='conv3',
            filters=(24, 48, 96), no_bias=no_bias, use_global_stats=use_global_stats, lr_mult=lr_mult)
    # inc3a
    inc3a = inception(conv3, prefix_group='inc3a',
            filters_1=96, filters_3=(16,64), filters_5=(16,32,32), no_bias=no_bias,
            use_global_stats=use_global_stats, do_pool=True, lr_mult=lr_mult)
    # inc3b
    inc3b = inception(inc3a, prefix_group='inc3b',
            filters_1=96, filters_3=(16,64), filters_5=(16,32,32), no_bias=no_bias,
            use_global_stats=use_global_stats, lr_mult=lr_mult)
    # inc3b/residual
    inc3b, inc3b_elt = residual_inc(conv3, inc3b, prefix_lhs='inc3a', prefix_rhs='inc3b', 
            num_filter=128, stride=(2,2), no_bias=no_bias, use_global_stats=use_global_stats, lr_mult=lr_mult)
    # inc3c
    inc3c = inception(inc3b, prefix_group='inc3c',
            filters_1=96, filters_3=(16,64), filters_5=(16,32,32), no_bias=no_bias,
            use_global_stats=use_global_stats, lr_mult=lr_mult)
    # inc3d
    inc3d = inception(inc3c, prefix_group='inc3d',
            filters_1=96, filters_3=(16,64), filters_5=(16,32,32), no_bias=no_bias,
            use_global_stats=use_global_stats, lr_mult=lr_mult)
    # inc3e
    inc3e = inception(inc3d, prefix_group='inc3e',
            filters_1=96, filters_3=(16,64), filters_5=(16,32,32), no_bias=no_bias,
            use_global_stats=use_global_stats, lr_mult=lr_mult)
    # inc3e/residual
    inc3e, _ = residual_inc(inc3b_elt, inc3e, prefix_lhs='inc3c', prefix_rhs='inc3e', 
            num_filter=128, stride=(1,1), no_bias=no_bias, use_global_stats=use_global_stats, lr_mult=lr_mult)
    # inc4a
    inc4a = inception(inc3e, prefix_group='inc4a',
            filters_1=128, filters_3=(32,96), filters_5=(16,32,32), no_bias=no_bias,
            use_global_stats=use_global_stats, do_pool=True, lr_mult=lr_mult)
    # inc4b
    inc4b = inception(inc4a, prefix_group='inc4b',
            filters_1=128, filters_3=(32,96), filters_5=(16,32,32), no_bias=no_bias,
            use_global_stats=use_global_stats, lr_mult=lr_mult)
    # inc4b/residual
    inc4b, inc4b_elt = residual_inc(inc3e, inc4b, prefix_lhs='inc4a', prefix_rhs='inc4b', 
            num_filter=192, stride=(2,2), no_bias=no_bias, use_global_stats=use_global_stats, lr_mult=lr_mult)
    # inc4c
    inc4c = inception(inc4b, prefix_group='inc4c',
            filters_1=128, filters_3=(32,96), filters_5=(16,32,32), no_bias=no_bias,
            use_global_stats=use_global_stats, lr_mult=lr_mult)
    # inc4d
    inc4d = inception(inc4c, prefix_group='inc4d',
            filters_1=128, filters_3=(32,96), filters_5=(16,32,32), no_bias=no_bias,
            use_global_stats=use_global_stats, lr_mult=lr_mult)
    # inc4e
    inc4e = inception(inc4d, prefix_group='inc4e',
            filters_1=128, filters_3=(32,96), filters_5=(16,32,32), no_bias=no_bias,
            use_global_stats=use_global_stats, lr_mult=lr_mult)
    # inc4e/residual
    inc4e, _ = residual_inc(inc4b_elt, inc4e, prefix_lhs='inc4c', prefix_rhs='inc4e', 
            num_filter=384, stride=(1,1), no_bias=no_bias, use_global_stats=use_global_stats, lr_mult=lr_mult)

    # hyperfeature
    downsample = mx.sym.Pooling(conv3, name='downsample', 
            kernel=(3,3), pad=(0,0), stride=(2,2), pool_type='max', pooling_convention='full')
    upsample = mx.sym.UpSampling(inc4e, name='upsample', scale=2, 
            sample_type='bilinear', num_filter=384, num_args=2)
    concat = mx.sym.concat(downsample, inc3e, upsample)

    # features for rpn and rcnn
    convf_rpn = convolution(concat, name='convf_rpn', 
            num_filter=128, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=no_bias, lr_mult=lr_mult)
    reluf_rpn = mx.sym.Activation(convf_rpn, name='reluf_rpn', act_type='relu')

    convf_2 = convolution(concat, name='convf_2', 
            num_filter=384, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=no_bias, lr_mult=lr_mult)
    reluf_2 = mx.sym.Activation(convf_2, name='reluf_2', act_type='relu')
    concat_convf = mx.sym.concat(reluf_rpn, reluf_2, name='concat_convf')

    return reluf_rpn, concat_convf


def get_pvanet_train(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS):
    ''' network for training '''
    # global parameters, (maybe will be changed)
    no_bias = False
    use_global_stats = True

    data = mx.sym.Variable(name='data')
    im_info = mx.sym.Variable(name='im_info')
    gt_boxes = mx.sym.Variable(name='gt_boxes')
    rpn_label = mx.sym.Variable(name='label')
    rpn_bbox_target = mx.sym.Variable(name='bbox_target')
    rpn_bbox_weight = mx.sym.Variable(name='bbox_weight')

    # shared conv layers
    reluf_rpn, concat_convf = \
            pvanet_preact(data, no_bias=no_bias, use_global_stats=use_global_stats, lr_mult=1.0)
    
    # RPN layers
    rpn_conv1 = convolution(reluf_rpn, name='rpn_conv1', 
            num_filter=384, pad=(1,1), kernel=(3,3), stride=(1,1))
    rpn_relu1 = mx.sym.Activation(rpn_conv1, name='rpn_relu1', act_type='relu')
    rpn_cls_score = convolution(rpn_relu1, name='rpn_cls_score', 
            num_filter=2*num_anchors, pad=(0,0), kernel=(1,1), stride=(1,1))
    rpn_bbox_pred = convolution(rpn_relu1, name='rpn_bbox_pred', 
            num_filter=4*num_anchors, pad=(0,0), kernel=(1,1), stride=(1,1))

    # prepare rpn data
    rpn_cls_score_reshape = mx.sym.Reshape(rpn_cls_score, name='rpn_cls_score_reshape', shape=(0, 2, -1, 0))

    # classification
    rpn_cls_prob = mx.sym.SoftmaxOutput(rpn_cls_score_reshape, label=rpn_label, multi_output=True,
            normalization='valid', use_ignore=True, ignore_label=-1, name='rpn_cls_prob', out_grad=False)
    # [hyunjoon] weighted loss
    rpn_cls_loss = mx.sym.Custom(rpn_cls_prob, rpn_label, op_type='softmax_loss', 
            ignore_label=-1, use_ignore=True, multi_output=True, normalization='valid')
    alpha_rpn_cls = mx.sym.var(name='rpn_cls_beta', shape=(1,), lr_mult=0.1, wd_mult=0.0)
    w_rpn_cls_loss = rpn_cls_loss * mx.sym.exp(-alpha_rpn_cls) + alpha_rpn_cls # * 1.0
    w_rpn_cls_loss = mx.sym.MakeLoss(w_rpn_cls_loss, name='rpn_cls_loss') 
    #         # grad_scale=1.0 / config.TRAIN.RPN_BATCH_SIZE)

    # bounding box regression
    rpn_bbox_loss_ = rpn_bbox_weight * \
            mx.sym.smooth_l1(rpn_bbox_pred - rpn_bbox_target, name='rpn_bbox_loss_', scalar=3.0)
    # [hyunjoon] weighted loss
    rpn_bbox_loss = mx.sym.sum(rpn_bbox_loss_) / config.TRAIN.RPN_BATCH_SIZE
    alpha_rpn_bbox = mx.sym.var(name='rpn_bbox_beta', shape=(1,), lr_mult=0.1, wd_mult=0.0)
    w_rpn_bbox_loss = rpn_bbox_loss * mx.sym.exp(-alpha_rpn_bbox) + alpha_rpn_bbox # * 1.0
    w_rpn_bbox_loss = mx.sym.MakeLoss(w_rpn_bbox_loss, name='rpn_bbox_loss') 
    # # ,grad_scale=1.0 / config.TRAIN.RPN_BATCH_SIZE)
    # rpn_bbox_loss = mx.sym.MakeLoss(rpn_bbox_loss_, name='rpn_bbox_loss', 
    #         grad_scale=1.0 / config.TRAIN.RPN_BATCH_SIZE)

    # ROI proposal
    rpn_cls_act = mx.sym.SoftmaxActivation(data=rpn_cls_score_reshape, mode='channel', name='rpn_cls_act')
    rpn_cls_act_reshape = mx.sym.Reshape(rpn_cls_act, name='rpn_cls_act_reshape',
            shape=(0, 2 * num_anchors, -1, 0))
    if config.TRAIN.CXX_PROPOSAL:
        rois = mx.contrib.symbol.Proposal(
            cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
            feature_stride=config.RPN_FEAT_STRIDE, 
            scales=tuple(config.ANCHOR_SCALES), 
            ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=config.TRAIN.RPN_PRE_NMS_TOP_N, 
            rpn_post_nms_top_n=config.TRAIN.RPN_POST_NMS_TOP_N,
            threshold=config.TRAIN.RPN_NMS_THRESH, 
            rpn_min_size=config.TRAIN.RPN_MIN_SIZE)
    else:
        rois = mx.sym.Custom(
            cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
            op_type='proposal', 
            feat_stride=config.RPN_FEAT_STRIDE,
            scales=tuple(config.ANCHOR_SCALES), 
            ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=config.TRAIN.RPN_PRE_NMS_TOP_N, 
            rpn_post_nms_top_n=config.TRAIN.RPN_POST_NMS_TOP_N,
            threshold=config.TRAIN.RPN_NMS_THRESH, 
            rpn_min_size=config.TRAIN.RPN_MIN_SIZE)

    # ROI proposal target
    gt_boxes_reshape = mx.sym.Reshape(data=gt_boxes, shape=(-1, 5), name='gt_boxes_reshape')
    group = mx.sym.Custom(rois=rois, gt_boxes=gt_boxes_reshape, op_type='proposal_target',
                             num_classes=num_classes, batch_images=config.TRAIN.BATCH_IMAGES,
                             batch_rois=config.TRAIN.BATCH_ROIS, fg_fraction=config.TRAIN.FG_FRACTION)
    rois = group[0]
    label = group[1]
    bbox_target = group[2]
    bbox_weight = group[3]

    # Fast R-CNN
    roi_pool = mx.sym.ROIPooling(concat_convf, name='roi_pool5', rois=rois, 
            pooled_size=(6, 6), spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)
    flat5 = mx.sym.Flatten(roi_pool, name='flat5')

    fc6 = fullyconnected(flat5, name='fc6', num_hidden=4096)
    fc6_bn = batchnorm(fc6, name='fc6/bn', use_global_stats=True, fix_gamma=False)
    fc6_dropout = mx.sym.Dropout(fc6_bn, name='fc6/dropout', p=0.25)
    fc6_relu = mx.sym.Activation(fc6_dropout, name='fc6/relu', act_type='relu')

    fc7 = fullyconnected(fc6_relu, name='fc7', num_hidden=4096)
    fc7_bn = batchnorm(fc7, name='fc7/bn', use_global_stats=True, fix_gamma=False)
    fc7_dropout = mx.sym.Dropout(fc7_bn, name='fc7/dropout', p=0.25)
    fc7_relu = mx.sym.Activation(fc7_dropout, name='fc7/relu', act_type='relu')

    # classification
    cls_score = fullyconnected(fc7_relu, name='cls_score', num_hidden=num_classes)
    cls_prob = mx.sym.SoftmaxOutput(data=cls_score, label=label, name='cls_prob', 
            normalization='batch', out_grad=True)
    # [hyunjoon] weighted loss
    cls_loss = mx.sym.Custom(cls_prob, label, op_type='softmax_loss', normalization='batch')
    alpha_cls = mx.sym.var(name='cls_beta', shape=(1,), lr_mult=0.1, wd_mult=0.0)
    w_cls_loss = cls_loss * mx.sym.exp(-alpha_cls) + alpha_cls # * 1.0
    w_cls_loss = mx.sym.MakeLoss(w_cls_loss, name='cls_loss') #, grad_scale=1.0 / config.TRAIN.BATCH_ROIS)
    # bounding box regression
    bbox_pred = fullyconnected(fc7_relu, name='bbox_pred', num_hidden=num_classes*4)
    bbox_loss_ = bbox_weight * \
            mx.sym.smooth_l1(bbox_pred - bbox_target, name='bbox_loss_', scalar=1.0)
    # [hyunjoon] weighted loss
    bbox_loss = mx.sym.sum(bbox_loss_) / config.TRAIN.BATCH_ROIS
    alpha_bbox = mx.sym.var(name='bbox_beta', shape=(1,), lr_mult=0.1, wd_mult=0.0)
    w_bbox_loss = bbox_loss * mx.sym.exp(-alpha_bbox) + alpha_bbox # * 1.0
    w_bbox_loss = mx.sym.MakeLoss(w_bbox_loss, name='bbox_loss') #, grad_scale=1.0 / config.TRAIN.BATCH_ROIS)
    # bbox_loss = mx.sym.MakeLoss(bbox_loss_, name='bbox_loss', grad_scale=1.0 / config.TRAIN.BATCH_ROIS)

    # reshape output
    label = mx.sym.Reshape(label, name='label_reshape', shape=(config.TRAIN.BATCH_IMAGES, -1))
    cls_prob = mx.sym.Reshape(cls_prob, name='cls_prob_reshape', 
            shape=(config.TRAIN.BATCH_IMAGES, -1, num_classes))
    bbox_loss = mx.sym.Reshape(bbox_loss, name='bbox_loss_reshape', 
            shape=(config.TRAIN.BATCH_IMAGES, -1, 4 * num_classes))

    group = mx.sym.Group([w_rpn_cls_loss, w_rpn_bbox_loss, mx.sym.BlockGrad(rpn_cls_prob), \
            w_cls_loss, w_bbox_loss, mx.sym.BlockGrad(cls_prob), mx.sym.BlockGrad(label)])
    # group = mx.sym.Group([rpn_cls_prob, rpn_bbox_loss,  \
    #         cls_prob, bbox_loss, mx.sym.BlockGrad(label)])
    return group


def get_pvanet_test(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS):
    ''' network for test '''
    data = mx.sym.Variable(name='data')
    im_info = mx.sym.Variable(name='im_info', init=mx.init.Zero())

    no_bias = False
    use_global_stats = True

    # shared conv layers
    reluf_rpn, concat_convf = pvanet_preact(data, no_bias=no_bias, use_global_stats=use_global_stats)
    
    # RPN layers
    rpn_conv1 = convolution(reluf_rpn, name='rpn_conv1', 
            num_filter=384, pad=(1,1), kernel=(3,3), stride=(1,1))
    rpn_relu1 = mx.sym.Activation(rpn_conv1, name='rpn_relu1', act_type='relu')
    rpn_cls_score = convolution(rpn_relu1, name='rpn_cls_score', 
            num_filter=2*num_anchors, pad=(0,0), kernel=(1,1), stride=(1,1))
    rpn_bbox_pred = convolution(rpn_relu1, name='rpn_bbox_pred', 
            num_filter=4*num_anchors, pad=(0,0), kernel=(1,1), stride=(1,1))

    # ROI Proposal
    rpn_cls_score_reshape = mx.sym.Reshape(
        data=rpn_cls_score, shape=(0, 2, -1, 0), name='rpn_cls_score_reshape')
    rpn_cls_prob = mx.sym.SoftmaxActivation(
        data=rpn_cls_score_reshape, mode='channel', name='rpn_cls_prob')
    rpn_cls_prob_reshape = mx.sym.Reshape(
        data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_prob_reshape')
    if config.TEST.CXX_PROPOSAL:
        rois = mx.contrib.symbol.Proposal(
            cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
            feature_stride=config.RPN_FEAT_STRIDE, 
            scales=tuple(config.ANCHOR_SCALES), 
            ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=config.TEST.RPN_PRE_NMS_TOP_N, 
            rpn_post_nms_top_n=config.TEST.RPN_POST_NMS_TOP_N,
            threshold=config.TEST.RPN_NMS_THRESH, 
            rpn_min_size=config.TEST.RPN_MIN_SIZE)
    else:
        rois = mx.sym.Custom(
            cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
            op_type='proposal', 
            feat_stride=config.RPN_FEAT_STRIDE,
            scales=tuple(config.ANCHOR_SCALES), 
            ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=config.TEST.RPN_PRE_NMS_TOP_N, 
            rpn_post_nms_top_n=config.TEST.RPN_POST_NMS_TOP_N,
            threshold=config.TEST.RPN_NMS_THRESH, 
            rpn_min_size=config.TEST.RPN_MIN_SIZE)

    # Fast R-CNN
    roi_pool = mx.sym.ROIPooling(concat_convf, name='roi_pool5', rois=rois, 
            pooled_size=(6, 6), spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)
    flat5 = mx.sym.Flatten(roi_pool, name='flat5')

    fc6 = fullyconnected(flat5, name='fc6', num_hidden=4096)
    fc6_bn = batchnorm(fc6, name='fc6/bn', use_global_stats=True, fix_gamma=False)
    fc6_relu = mx.sym.Activation(fc6_bn, name='fc6/relu', act_type='relu')

    fc7 = fullyconnected(fc6_relu, name='fc7', num_hidden=4096)
    fc7_bn = batchnorm(fc7, name='fc7/bn', use_global_stats=True, fix_gamma=False)
    fc7_relu = mx.sym.Activation(fc7_bn, name='fc7/relu', act_type='relu')

    # classification
    cls_score = fullyconnected(fc7_relu, name='cls_score', num_hidden=num_classes)
    cls_prob = mx.sym.SoftmaxActivation(cls_score, name='cls_prob')
    # bounding box regression
    bbox_pred = fullyconnected(fc7_relu, name='bbox_pred', num_hidden=num_classes*4)

    # reshape output
    cls_prob = mx.sym.Reshape(cls_prob, name='cls_prob_reshape', 
            shape=(config.TEST.BATCH_IMAGES, -1, num_classes))
    bbox_pred = mx.sym.Reshape(bbox_pred, name='bbox_pred_reshape', 
            shape=(config.TEST.BATCH_IMAGES, -1, 4 * num_classes))

    # group output
    group = mx.sym.Group([rois, cls_prob, bbox_pred])
    return group
