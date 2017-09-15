import mxnet as mx
import proposal
import proposal_target_mpii
from rcnn.config import config


def conv_bn_relu(data, group_name,
        num_filter, kernel, pad, stride, no_bias,
        use_global_stats=True, use_crelu=False,
        get_syms=False):
    ''' use in mCReLU '''
    conv_name = group_name + ''
    concat_name = group_name + '/concat'
    bn_name = group_name + '/bn'
    relu_name = group_name + '/relu'

    syms = {}
    conv = mx.sym.Convolution(data, name=conv_name,
            num_filter=num_filter, kernel=kernel, pad=pad, stride=stride, no_bias=no_bias)
    syms['conv'] = conv
    if use_crelu:
        conv = mx.sym.concat(conv, -conv, name=concat_name)
        syms['concat'] = conv
    bn = mx.sym.BatchNorm(conv, name=bn_name, use_global_stats=use_global_stats, fix_gamma=False)
    syms['bn'] = bn
    relu = mx.sym.Activation(bn, name=relu_name, act_type='relu')
    if get_syms:
        return relu, syms
    else:
        return relu


def mcrelu(data, prefix_group, filters, no_bias, use_global_stats):
    ''' conv2 and conv3 '''
    group1 = conv_bn_relu(data, group_name=prefix_group+'_1',
            num_filter=filters[0], kernel=(1,1), pad=(0,0), stride=(1,1), no_bias=no_bias,
            use_global_stats=use_global_stats)
    group2 = conv_bn_relu(group1, group_name=prefix_group+'_2',
            num_filter=filters[1], kernel=(3,3), pad=(1,1), stride=(2,2), no_bias=no_bias,
            use_global_stats=use_global_stats, use_crelu=True)
    conv3 = mx.sym.Convolution(group2, name=prefix_group+'_3/out',
            num_filter=filters[2], kernel=(1,1), pad=(0,0), stride=(1,1), no_bias=no_bias)
    proj1 = mx.sym.Convolution(data, name=prefix_group+'_1/proj',
            num_filter=filters[2], kernel=(1,1), pad=(0,0), stride=(2,2), no_bias=no_bias)
    bn3 = mx.sym.BatchNorm(conv3+proj1, name=prefix_group+'_3/elt/bn',
            use_global_stats=use_global_stats, fix_gamma=False)
    relu3 = mx.sym.Activation(bn3, name=prefix_group+'_3/relu', act_type='relu')
    return relu3


def inception(data, prefix_group,
        filters_1, filters_3, filters_5, no_bias,
        use_global_stats, do_pool=False):
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
            use_global_stats=use_global_stats)
    # conv3
    conv3_1 = conv_bn_relu(data, group_name=prefix_group+'/conv3_1',
            num_filter=filters_3[0], kernel=(1,1), pad=(0,0), stride=(1,1), no_bias=no_bias,
            use_global_stats=use_global_stats)
    conv3_2 = conv_bn_relu(conv3_1, group_name=prefix_group+'/conv3_2',
            num_filter=filters_3[1], kernel=(3,3), pad=(1,1), stride=ss, no_bias=no_bias,
            use_global_stats=use_global_stats)
    # conv5
    conv5_1 = conv_bn_relu(data, group_name=prefix_group+'/conv5_1',
            num_filter=filters_5[0], kernel=(1,1), pad=(0,0), stride=(1,1), no_bias=no_bias,
            use_global_stats=use_global_stats)
    conv5_2 = conv_bn_relu(conv5_1, group_name=prefix_group+'/conv5_2',
            num_filter=filters_5[1], kernel=(3,3), pad=(1,1), stride=(1,1), no_bias=no_bias,
            use_global_stats=use_global_stats)
    conv5_3 = conv_bn_relu(conv5_2, group_name=prefix_group+'/conv5_3',
            num_filter=filters_5[2], kernel=(3,3), pad=(1,1), stride=ss, no_bias=no_bias,
            use_global_stats=use_global_stats)
    return mx.sym.concat(conv1, conv3_2, conv5_3)


def residual_inc(lhs, rhs, prefix_lhs, prefix_rhs, num_filter, stride, no_bias, use_global_stats):
    ''' residual connection between inception layers '''
    lhs = mx.sym.Convolution(lhs, name=prefix_lhs+'/proj',
            num_filter=num_filter, kernel=(1,1), pad=(0,0), stride=stride, no_bias=no_bias)
    rhs = mx.sym.Convolution(rhs, name=prefix_rhs+'/out',
            num_filter=num_filter, kernel=(1,1), pad=(0,0), stride=(1,1), no_bias=no_bias)
    elt = lhs+rhs
    bn = mx.sym.BatchNorm(elt, name=prefix_rhs+'/elt/bn',
            use_global_stats=use_global_stats, fix_gamma=False)
    relu = mx.sym.Activation(bn, act_type='relu')
    return relu
    # return relu, elt


def pvanet_preact(data, use_global_stats=True, no_bias=False):
    ''' pvanet 10.0 '''
    conv1 = conv_bn_relu(data, group_name='conv1',
            num_filter=16, kernel=(4,4), pad=(1,1), stride=(2,2), no_bias=no_bias,
            use_global_stats=use_global_stats, use_crelu=True)
    # conv2
    conv2 = mcrelu(conv1, prefix_group='conv2',
            filters=(16, 24, 48), no_bias=no_bias, use_global_stats=use_global_stats)
    # conv3
    conv3 = mcrelu(conv2, prefix_group='conv3',
            filters=(24, 48, 96), no_bias=no_bias, use_global_stats=use_global_stats)
    # inc3a
    inc3a = inception(conv3, prefix_group='inc3a',
            filters_1=96, filters_3=(16,64), filters_5=(16,32,32), no_bias=no_bias,
            use_global_stats=use_global_stats, do_pool=True)
    # inc3b
    inc3b = inception(inc3a, prefix_group='inc3b',
            filters_1=96, filters_3=(16,64), filters_5=(16,32,32), no_bias=no_bias,
            use_global_stats=use_global_stats)
    # inc3b/residual
    inc3b = residual_inc(conv3, inc3b, prefix_lhs='inc3a', prefix_rhs='inc3b',
            num_filter=128, stride=(2,2), no_bias=no_bias, use_global_stats=use_global_stats)
    # inc3c
    inc3c = inception(inc3b, prefix_group='inc3c',
            filters_1=96, filters_3=(16,64), filters_5=(16,32,32), no_bias=no_bias,
            use_global_stats=use_global_stats)
    # inc3d
    inc3d = inception(inc3c, prefix_group='inc3d',
            filters_1=96, filters_3=(16,64), filters_5=(16,32,32), no_bias=no_bias,
            use_global_stats=use_global_stats)
    # inc3e
    inc3e = inception(inc3d, prefix_group='inc3e',
            filters_1=96, filters_3=(16,64), filters_5=(16,32,32), no_bias=no_bias,
            use_global_stats=use_global_stats)
    # inc3e/residual
    inc3e = residual_inc(inc3b, inc3e, prefix_lhs='inc3c', prefix_rhs='inc3e',
            num_filter=128, stride=(1,1), no_bias=no_bias, use_global_stats=use_global_stats)
    # inc4a
    inc4a = inception(inc3e, prefix_group='inc4a',
            filters_1=128, filters_3=(32,96), filters_5=(16,32,32), no_bias=no_bias,
            use_global_stats=use_global_stats, do_pool=True)
    # inc4b
    inc4b = inception(inc4a, prefix_group='inc4b',
            filters_1=128, filters_3=(32,96), filters_5=(16,32,32), no_bias=no_bias,
            use_global_stats=use_global_stats)
    # inc4b/residual
    inc4b = residual_inc(inc3e, inc4b, prefix_lhs='inc4a', prefix_rhs='inc4b',
            num_filter=192, stride=(2,2), no_bias=no_bias, use_global_stats=use_global_stats)
    # inc4c
    inc4c = inception(inc4b, prefix_group='inc4c',
            filters_1=128, filters_3=(32,96), filters_5=(16,32,32), no_bias=no_bias,
            use_global_stats=use_global_stats)
    # inc4d
    inc4d = inception(inc4c, prefix_group='inc4d',
            filters_1=128, filters_3=(32,96), filters_5=(16,32,32), no_bias=no_bias,
            use_global_stats=use_global_stats)
    # inc4e
    inc4e = inception(inc4d, prefix_group='inc4e',
            filters_1=128, filters_3=(32,96), filters_5=(16,32,32), no_bias=no_bias,
            use_global_stats=use_global_stats)
    # inc4e/residual
    inc4e = residual_inc(inc4b, inc4e, prefix_lhs='inc4c', prefix_rhs='inc4e',
            num_filter=384, stride=(1,1), no_bias=no_bias, use_global_stats=use_global_stats)

    # hyperfeature
    downsample = mx.sym.Pooling(conv3, name='downsample',
            kernel=(3,3), pad=(0,0), stride=(2,2), pool_type='max', pooling_convention='full')
    upsample = mx.sym.UpSampling(inc4e, name='upsample', scale=2,
            sample_type='bilinear', num_filter=384, num_args=2)
    concat = mx.sym.concat(downsample, inc3e, upsample)

    # features for rpn and rcnn
    convf_rpn = mx.sym.Convolution(concat, name='convf_rpn',
            num_filter=128, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=no_bias)
    reluf_rpn = mx.sym.Activation(convf_rpn, name='reluf_rpn', act_type='relu')

    convf_2 = mx.sym.Convolution(concat, name='convf_2',
            num_filter=384, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=no_bias)
    reluf_2 = mx.sym.Activation(convf_2, name='reluf_2', act_type='relu')
    concat_convf = mx.sym.concat(reluf_rpn, reluf_2, name='concat_convf')

    return reluf_rpn, concat_convf


def get_pvanet_mpii_train(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS):
    ''' network for training '''
    # global parameters, (maybe will be changed)
    no_bias = False
    use_global_stats = True

    data = mx.sym.Variable(name='data')
    im_info = mx.sym.Variable(name='im_info')
    gt_boxes = mx.sym.Variable(name='gt_boxes')
    gt_head_boxes = mx.sym.Variable(name='gt_head_boxes')
    gt_joints = mx.sym.Variable(name='gt_joints')
    rpn_label = mx.sym.Variable(name='label')
    rpn_bbox_target = mx.sym.Variable(name='bbox_target')
    rpn_bbox_weight = mx.sym.Variable(name='bbox_weight')

    # shared conv layers
    reluf_rpn, concat_convf = pvanet_preact(data, no_bias=no_bias, use_global_stats=use_global_stats)

    # RPN layers
    rpn_conv1 = mx.sym.Convolution(reluf_rpn, name='rpn_conv1',
            num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1))
    rpn_relu1 = mx.sym.Activation(rpn_conv1, name='rpn_relu1', act_type='relu')
    rpn_cls_score = mx.sym.Convolution(rpn_relu1, name='rpn_cls_score',
            num_filter=2*num_anchors, pad=(0,0), kernel=(1,1), stride=(1,1))
    rpn_bbox_pred = mx.sym.Convolution(rpn_relu1, name='rpn_bbox_pred',
            num_filter=4*num_anchors, pad=(0,0), kernel=(1,1), stride=(1,1))

    # prepare rpn data
    rpn_cls_score_reshape = mx.sym.Reshape(rpn_cls_score, name='rpn_cls_score_reshape', shape=(0, 2, -1, 0))

    # classification
    rpn_cls_prob = mx.sym.SoftmaxOutput(rpn_cls_score_reshape, label=rpn_label, multi_output=True,
            normalization='valid', use_ignore=True, ignore_label=-1, name='rpn_cls_prob')
    # bounding box regression
    rpn_bbox_loss_ = rpn_bbox_weight * \
            mx.sym.smooth_l1(rpn_bbox_pred - rpn_bbox_target, name='rpn_bbox_loss_', scalar=3.0)
    rpn_bbox_loss = mx.sym.MakeLoss(rpn_bbox_loss_, name='rpn_bbox_loss',
            grad_scale=1.0 / config.TRAIN.RPN_BATCH_SIZE)

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
    gt_head_boxes_reshape = mx.sym.Reshape(data=gt_head_boxes, shape=(-1, 4), name='gt_head_boxes_reshape')
    gt_joints_reshape = mx.sym.Reshape(data=gt_joints, shape=(-1, 12), name='gt_joints_reshape')
    group = mx.sym.Custom(rois=rois, gt_boxes=gt_boxes_reshape,
                          gt_head_boxes=gt_head_boxes_reshape, gt_joints=gt_joints_reshape,
                          op_type='proposal_target',
                          num_classes=num_classes, batch_images=config.TRAIN.BATCH_IMAGES,
                          batch_rois=config.TRAIN.BATCH_ROIS, fg_fraction=config.TRAIN.FG_FRACTION)
    rois = group[0]
    label = group[1]
    bbox_target = group[2]
    bbox_weight = group[3]

    head_gid = group[4]
    head_target = group[5]
    head_weight = group[6]

    joint_gid = group[7]
    joint_target = group[8]
    joint_weight = group[9]

    # Fast R-CNN
    roi_pool = mx.sym.ROIPooling(concat_convf, name='roi_pool5', rois=rois,
            pooled_size=(6, 6), spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)
    flat5 = mx.sym.Flatten(roi_pool, name='flat5')

    fc6 = mx.sym.FullyConnected(flat5, name='fc6', num_hidden=4096)
    fc6_bn = mx.sym.BatchNorm(fc6, name='fc6/bn', use_global_stats=True, fix_gamma=False)
    fc6_dropout = mx.sym.Dropout(fc6_bn, name='fc6/dropout', p=0.25)
    fc6_relu = mx.sym.Activation(fc6_dropout, name='fc6/relu', act_type='relu')

    fc7 = mx.sym.FullyConnected(fc6_relu, name='fc7', num_hidden=4096)
    fc7_bn = mx.sym.BatchNorm(fc7, name='fc7/bn', use_global_stats=True, fix_gamma=False)
    fc7_dropout = mx.sym.Dropout(fc7_bn, name='fc7/dropout', p=0.25)
    fc7_relu = mx.sym.Activation(fc7_dropout, name='fc7/relu', act_type='relu')

    # classification
    cls_score = mx.sym.FullyConnected(fc7_relu, name='cls_score', num_hidden=num_classes)
    cls_prob = mx.sym.SoftmaxOutput(data=cls_score, label=label, name='cls_prob', normalization='batch')
    # bounding box regression
    bbox_pred = mx.sym.FullyConnected(fc7_relu, name='bbox_pred', num_hidden=num_classes*4)
    bbox_loss_ = bbox_weight * \
            mx.sym.smooth_l1(bbox_pred - bbox_target, name='bbox_loss_', scalar=1.0)
    bbox_loss = mx.sym.MakeLoss(bbox_loss_, name='bbox_loss', grad_scale=1.0 / config.TRAIN.BATCH_ROIS)

    # head classification
    num_grid = config.PART_GRID_HW[0] * config.PART_GRID_HW[1] # including bg
    head_score = mx.sym.FullyConnected(fc7_relu, name='head_score', num_hidden=num_grid)
    head_prob = mx.sym.SoftmaxOutput(head_score, label=head_gid, name='head_prob', normalization='batch',
            use_ignore=True, ignore_label=-1)
    head_bbox_pred = mx.sym.FullyConnected(fc7_relu, name='head_pred', num_hidden=num_grid*4)
    head_bbox_loss_ = head_weight * \
            mx.sym.smooth_l1(head_bbox_pred - head_target, name='bbox_loss_', scalar=1.0)
    head_bbox_loss = mx.sym.MakeLoss(head_bbox_loss_, name='head_bbox_loss', grad_scale=0.2 / config.TRAIN.BATCH_ROIS)

    # joint classification
    joint_score = mx.sym.FullyConnected(fc7_relu, name='joint_score', num_hidden=num_grid*4)
    joint_probs = []
    joint_gids = []
    for i in range(4):
        sidx = i * num_grid
        eidx = (i+1) * num_grid
        scorei = mx.sym.slice_axis(joint_score, axis=1, begin=sidx, end=eidx)
        labeli = mx.sym.slice_axis(joint_gid, axis=1, begin=i, end=i+1)
        labeli = mx.sym.reshape(labeli, (-1,))
        joint_gids.append(labeli)
        joint_probs.append(mx.sym.SoftmaxOutput( \
                scorei, labeli, name='joint_prob{}'.format(i), normalization='batch',
                use_ignore=True, ignore_label=-1))
    joint_pred = mx.sym.FullyConnected(fc7_relu, name='joint_pred', num_hidden=num_grid*2*4)
    joint_loss_ = joint_weight * \
            mx.sym.smooth_l1(joint_pred - joint_target, name='joint_loss_', scalar=1.0)
    joint_losses = []
    for i in range(4):
        sidx = num_grid * i * 2
        eidx = num_grid * (i+1) * 2
        lossi = mx.sym.slice_axis(joint_loss_, axis=1, begin=sidx, end=eidx)
        joint_losses.append(mx.sym.MakeLoss( \
                lossi, name='joint_loss{}'.format(i), grad_scale=0.2 / config.TRAIN.BATCH_ROIS))

    # reshape output
    label = mx.sym.Reshape(label, name='label_reshape', shape=(config.TRAIN.BATCH_IMAGES, -1))
    cls_prob = mx.sym.Reshape(cls_prob, name='cls_prob_reshape',
            shape=(config.TRAIN.BATCH_IMAGES, -1, num_classes))
    bbox_loss = mx.sym.Reshape(bbox_loss, name='bbox_loss_reshape',
            shape=(config.TRAIN.BATCH_IMAGES, -1, 4 * num_classes))
    head_gid = mx.sym.Reshape(head_gid, name='head_gid_reshape', shape=(config.TRAIN.BATCH_IMAGES, -1))
    head_prob = mx.sym.Reshape(head_prob, name='head_prob_reshape',
            shape=(config.TRAIN.BATCH_IMAGES, -1, num_grid))
    head_bbox_loss = mx.sym.Reshape(head_bbox_loss, name='head_bbox_loss_reshape',
            shape=(config.TRAIN.BATCH_IMAGES, -1, num_grid*4))
    for i in range(4):
        joint_gids[i] = mx.sym.reshape(joint_gids[i], name='joint_gid{}_reshape'.format(i),
                shape=(config.TRAIN.BATCH_IMAGES, -1))
        joint_gids[i] = mx.sym.BlockGrad(joint_gids[i])
        joint_probs[i] = mx.sym.reshape(joint_probs[i], name='joint_prob{}_reshape'.format(i),
                shape=(config.TRAIN.BATCH_IMAGES, -1, num_grid))
        joint_losses[i] = mx.sym.reshape(joint_losses[i], name='joint_loss{}_reshape'.format(i),
                shape=(config.TRAIN.BATCH_IMAGES, -1, num_grid*2))

    loss_group = [rpn_cls_prob, rpn_bbox_loss, cls_prob, bbox_loss, mx.sym.BlockGrad(label)]
    loss_group += [head_prob, head_bbox_loss, mx.sym.BlockGrad(head_gid)]
    loss_group += joint_probs
    loss_group += joint_losses
    loss_group += joint_gids

    return mx.sym.Group(loss_group)


def get_pvanet_mpii_test(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS):
    ''' network for test '''
    data = mx.sym.Variable(name='data')
    im_info = mx.sym.Variable(name='im_info', init=mx.init.Zero())

    no_bias = False
    use_global_stats = True

    # shared conv layers
    reluf_rpn, concat_convf = pvanet_preact(data, no_bias=no_bias, use_global_stats=use_global_stats)

    # RPN layers
    rpn_conv1 = mx.sym.Convolution(reluf_rpn, name='rpn_conv1',
            num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1))
    rpn_relu1 = mx.sym.Activation(rpn_conv1, name='rpn_relu1', act_type='relu')
    rpn_cls_score = mx.sym.Convolution(rpn_relu1, name='rpn_cls_score',
            num_filter=2*num_anchors, pad=(0,0), kernel=(1,1), stride=(1,1))
    rpn_bbox_pred = mx.sym.Convolution(rpn_relu1, name='rpn_bbox_pred',
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

    fc6 = mx.sym.FullyConnected(flat5, name='fc6', num_hidden=4096)
    fc6_bn = mx.sym.BatchNorm(fc6, name='fc6/bn', use_global_stats=True, fix_gamma=False)
    fc6_relu = mx.sym.Activation(fc6_bn, name='fc6/relu', act_type='relu')

    fc7 = mx.sym.FullyConnected(fc6_relu, name='fc7', num_hidden=4096)
    fc7_bn = mx.sym.BatchNorm(fc7, name='fc7/bn', use_global_stats=True, fix_gamma=False)
    fc7_relu = mx.sym.Activation(fc7_bn, name='fc7/relu', act_type='relu')

    # classification
    cls_score = mx.sym.FullyConnected(fc7_relu, name='cls_score', num_hidden=num_classes)
    cls_prob = mx.sym.SoftmaxActivation(cls_score, name='cls_prob')
    # bounding box regression
    bbox_pred = mx.sym.FullyConnected(fc7_relu, name='bbox_pred', num_hidden=num_classes*4)

    # head classification
    num_grid = config.PART_GRID_HW[0] * config.PART_GRID_HW[1] # including bg
    head_score = mx.sym.FullyConnected(fc7_relu, name='head_score', num_hidden=num_grid)
    head_prob = mx.sym.SoftmaxActivation(head_score, name='head_prob')
    head_bbox_pred = mx.sym.FullyConnected(fc7_relu, name='head_pred', num_hidden=num_grid*4)

    # joint classification
    joint_score = mx.sym.FullyConnected(fc7_relu, name='joint_score', num_hidden=num_grid*4)
    joint_pred = mx.sym.FullyConnected(fc7_relu, name='joint_pred', num_hidden=num_grid*2*4)
    joint_probs = []
    joint_preds = []
    for i in range(4):
        sidx = i * num_grid
        eidx = (i+1) * num_grid
        scorei = mx.sym.slice_axis(joint_score, axis=1, begin=sidx, end=eidx)
        joint_probs.append(mx.sym.SoftmaxActivation(scorei, name='joint_prob{}'.format(i)))
        joint_preds.append(mx.sym.slice_axis(joint_pred, axis=1, begin=sidx*2, end=eidx*2))

    # reshape output
    cls_prob = mx.sym.Reshape(cls_prob, name='cls_prob_reshape',
            shape=(config.TEST.BATCH_IMAGES, -1, num_classes))
    bbox_pred = mx.sym.Reshape(bbox_pred, name='bbox_pred_reshape',
            shape=(config.TEST.BATCH_IMAGES, -1, 4 * num_classes))

    head_prob = mx.sym.reshape(head_prob, name='head_prob_reshape',
            shape=(config.TEST.BATCH_IMAGES, -1, num_grid))
    head_pred = mx.sym.Reshape(head_bbox_pred, name='head_pred_reshape',
            shape=(config.TEST.BATCH_IMAGES, -1, 4 * num_grid))

    for i in range(4):
        joint_probs[i] = mx.sym.reshape(joint_probs[i], name='joint_prob{}_reshape'.format(i),
                shape=(config.TEST.BATCH_IMAGES, -1, num_grid))
        joint_preds[i] = mx.sym.reshape(joint_preds[i], name='joint_pred{}_reshape'.format(i),
                shape=(config.TEST.BATCH_IMAGES, -1, 2 * num_grid))

    # group output
    group = [rois, cls_prob, bbox_pred, head_prob, head_pred]
    group += joint_probs
    group += joint_preds
    return mx.sym.Group(group)
