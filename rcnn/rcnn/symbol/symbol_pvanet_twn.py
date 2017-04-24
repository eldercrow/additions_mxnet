import mxnet as mx
import proposal
import proposal_target
from rcnn.config import config
from ternarize import *

def conv_twn(data, shape, name=None, pad=(0,0), stride=(1,1), no_bias=False):
    ''' ternary weight convolution '''
    conv_weight = mx.sym.var(name=name+'_weight', shape=shape, attr={'__wd_mult__': '0.0'}, dtype='float32')
    weight = mx.sym.Custom(conv_weight, op_type='ternarize', soft_ternarize=False)
    conv = mx.sym.Convolution(data=data, weight=weight, name=name, num_filter=shape[0], 
            kernel=shape[2:], pad=pad, stride=stride, no_bias=no_bias)
    return conv

def fc_twn(data, shape, name=None, no_bias=False):
    ''' ternary weight fc '''
    fc_weight = mx.sym.var(name=name+'_weight', shape=shape, attr={'__wd_mult__': '0.0'}, dtype='float32')
    # weight, alpha = mx.sym.Ternarize(fc_weight)
    # weight = mx.sym.broadcast_mul(weight, alpha)
    weight = mx.sym.Custom(fc_weight, op_type='ternarize', soft_ternarize=False)
    fc = mx.sym.FullyConnected(data=data, weight=weight, name=name, num_hidden=shape[0], no_bias=no_bias)
    return fc

def conv_bn_relu(data, group_name, shape, pad, stride, use_global):
    """ used in mCReLU """
    bn_name = group_name + '_bn'
    relu_name = group_name + '_relu'
    conv_name = group_name + '_conv'

    conv = conv_twn(data, shape, name=conv_name, 
            pad=pad, stride=stride, no_bias=True)
    # conv = mx.symbol.Convolution(name=conv_name, data=data, 
    #         num_filter=num_filter, pad=pad, kernel=kernel, stride=stride, no_bias=True)
    bn = mx.symbol.BatchNorm(name=bn_name, data=conv, use_global_stats=use_global, fix_gamma=False)
    relu = mx.symbol.Activation(name=relu_name, data=bn, act_type='relu')

    return relu

def bn_relu_conv(data, group_name, shape, pad, stride, use_global):
    """ used in mCReLU """
    bn_name = group_name + '_bn'
    relu_name = group_name + '_relu'
    conv_name = group_name + '_conv'

    bn = mx.symbol.BatchNorm(name=bn_name, data=data, use_global_stats=use_global, fix_gamma=False)
    relu = mx.symbol.Activation(name=relu_name, data=bn, act_type='relu')
    conv = conv_twn(relu, shape, name=conv_name, 
            pad=pad, stride=stride, no_bias=False)
    # conv = mx.symbol.Convolution(name=conv_name, data=relu, 
    #         num_filter=num_filter, pad=pad, kernel=kernel, stride=stride, no_bias=False)
    return conv

def bn_crelu_conv(data, group_name, shape, pad, stride, use_global):
    """ used in mCReLU """
    concat_name = group_name + '_concat'
    bn_name = group_name + '_bn'
    relu_name = group_name + '_relu'
    conv_name = group_name + '_conv'

    concat = mx.symbol.Concat(name=concat_name, *[data, -data])
    bn = mx.symbol.BatchNorm(name=bn_name, data=concat, use_global_stats=use_global, fix_gamma=False)
    relu = mx.symbol.Activation(name=relu_name, data=bn, act_type='relu')
    conv = conv_twn(relu, shape, name=conv_name, 
            pad=pad, stride=stride, no_bias=False)
    # conv = mx.symbol.Convolution(name=conv_name, data=relu, 
    #         num_filter=num_filter, pad=pad, kernel=kernel, stride=stride, no_bias=False)
    return conv

def mCReLU(data, group_name, filters, strides, use_global, n_curr_ch):
    """ 
    """
    kernels = ((1,1), (3,3), (1,1))
    pads = ((0,0), (1,1), (0,0))

    shapes = []
    shapes.append((filters[0], n_curr_ch, 1, 1))
    shapes.append((filters[1], filters[0], 3, 3))
    shapes.append((filters[2], filters[1]*2, 1, 1))

    conv1 = bn_relu_conv(data=data, group_name=group_name+'_1', 
            shape=shapes[0], pad=pads[0], stride=strides[0], use_global=use_global)
    conv2 = bn_relu_conv(data=conv1, group_name=group_name+'_2', 
            shape=shapes[1], pad=pads[1], stride=strides[1], use_global=use_global)
    conv3 = bn_crelu_conv(data=conv2, group_name=group_name+'_3', 
            shape=shapes[2], pad=pads[2], stride=strides[2], use_global=use_global)
    # conv1 = bn_relu_conv(data=data, group_name=group_name+'_1', 
    #         num_filter=filters[0], pad=pads[0], kernel=kernels[0], stride=strides[0], use_global=use_global)
    # conv2 = bn_relu_conv(data=conv1, group_name=group_name+'_2', 
    #         num_filter=filters[1], pad=pads[1], kernel=kernels[1], stride=strides[1], use_global=use_global)
    # conv3 = bn_crelu_conv(data=conv2, group_name=group_name+'_3', 
    #         num_filter=filters[2], pad=pads[2], kernel=kernels[2], stride=strides[2], use_global=use_global)

    ss = 1
    for s in strides:
        ss *= s[0]
    need_proj = (n_curr_ch != filters[2]) or (ss != 1)
    if need_proj:
        proj = conv_twn(data, shape=(filters[2], n_curr_ch, 1, 1), name=group_name+'_proj', 
                pad=(0,0), stride=(ss,ss))
        res = conv3 + proj
    else:
        res = conv3 + data

    return res, filters[2]

# final_bn is to handle stupid redundancy in the original model
def inception(data, group_name, 
        filter_1, filters_3, filters_5, filter_p, filter_out, stride, use_global, n_curr_ch, final_bn=False):
    """
    """
    group_name = group_name + '_incep'

    group_name_1 = group_name + '_0'
    group_name_3 = group_name + '_1'
    group_name_5 = group_name + '_2'

    incep_bn = mx.symbol.BatchNorm(name=group_name+'_bn', data=data, 
        use_global_stats=use_global, fix_gamma=False)
    incep_relu = mx.symbol.Activation(name=group_name+'_relu', data=incep_bn, act_type='relu')

    incep_0 = conv_bn_relu(data=incep_relu, group_name=group_name_1, 
            shape=(filter_1, n_curr_ch, 1, 1), pad=(0,0), stride=stride, use_global=use_global)

    incep_1_reduce = conv_bn_relu(data=incep_relu, group_name=group_name_3+'_reduce', 
            shape=(filters_3[0], n_curr_ch, 1, 1), pad=(0,0), stride=stride, use_global=use_global)
    incep_1_0 = conv_bn_relu(data=incep_1_reduce, group_name=group_name_3+'_0', 
            shape=(filters_3[1], filters_3[0], 3, 3), pad=(1,1), stride=(1,1), use_global=use_global)

    incep_2_reduce = conv_bn_relu(data=incep_relu, group_name=group_name_5+'_reduce', 
            shape=(filters_5[0], n_curr_ch, 1, 1), pad=(0,0), stride=stride, use_global=use_global)
    incep_2_0 = conv_bn_relu(data=incep_2_reduce, group_name=group_name_5+'_0', 
            shape=(filters_5[1], filters_5[0], 3, 3), pad=(1,1), stride=(1,1), use_global=use_global)
    incep_2_1 = conv_bn_relu(data=incep_2_0, group_name=group_name_5+'_1', 
            shape=(filters_5[2], filters_5[1], 3, 3), pad=(1,1), stride=(1,1), use_global=use_global)

    incep_layers = [incep_0, incep_1_0, incep_2_1]
    nch_incep = filter_1 + filters_3[1] + filters_5[2]

    if filter_p is not None:
        incep_p_pool = mx.symbol.Pooling(name=group_name+'_pool', data=incep_relu, pooling_convention='full', 
                pad=(0,0), kernel=(3,3), stride=(2,2), pool_type='max')
        incep_p_proj = conv_bn_relu(data=incep_p_pool, group_name=group_name+'_poolproj', 
                shape=(filter_p, n_curr_ch, 1, 1), pad=(0,0), stride=(1,1), use_global=use_global)
        incep_layers.append(incep_p_proj)
        nch_incep += filter_p

    incep = mx.symbol.Concat(name=group_name, *incep_layers)
    out_conv = conv_twn(data=incep, name=group_name.replace('_incep', '_out_conv'), 
            shape=(filter_out, nch_incep, 1, 1), pad=(0, 0))

    # final_bn is to handle stupid redundancy in the original model
    if final_bn:
        out_conv = mx.symbol.BatchNorm(name=group_name.replace('_incep', '_out_bn'), data=out_conv, 
                use_global_stats=use_global, fix_gamma=False)
    
    if n_curr_ch != filter_out or stride[0] > 1:
        out_proj = conv_twn(data, name=group_name.replace('_incep', '_proj'),
                shape=(filter_out, n_curr_ch, 1, 1), stride=stride, pad=(0, 0))
        # out_proj = mx.symbol.Convolution(name=group_name.replace('_incep', '_proj'), data=data, 
        #         num_filter=filter_out, kernel=(1,1), stride=stride, pad=(0,0))
        res = out_conv + out_proj
    else:
        res = out_conv + data

    return res, filter_out

def pvanet_preact(data, is_test):
    """ PVANet 9.0 """
    conv1_1_conv = mx.symbol.Convolution(name='conv1_1_conv', data=data, 
            num_filter=16, pad=(3,3), kernel=(7,7), stride=(2,2), no_bias=True)
    conv1_1_concat = mx.symbol.Concat(name='conv1_1_concat', *[conv1_1_conv, -conv1_1_conv])
    conv1_1_bn = mx.symbol.BatchNorm(name='conv1_1_bn', data=conv1_1_concat, 
            use_global_stats=is_test, fix_gamma=False)
    conv1_1_relu = mx.symbol.Activation(name='conv1_1_relu', data=conv1_1_bn, act_type='relu')
    pool1 = mx.symbol.Pooling(name='pool1', data=conv1_1_relu, 
            pooling_convention='full', pad=(0,0), kernel=(3,3), stride=(2,2), pool_type='max')
    
    # no pre bn-scale-relu for 2_1_1
    conv2_1_1_conv = conv_twn(pool1, name='conv2_1_1_conv', shape=(24, 32, 1, 1), pad=(0, 0))
    conv2_1_2_conv = bn_relu_conv(data=conv2_1_1_conv, group_name='conv2_1_2', 
            shape=(24, 24, 3, 3), pad=(1,1), stride=(1,1), use_global=is_test)
    conv2_1_3_conv = bn_crelu_conv(data=conv2_1_2_conv, group_name='conv2_1_3', 
            shape=(64, 48, 1, 1), pad=(0,0), stride=(1,1), use_global=is_test)
    conv2_1_proj = conv_twn(pool1, name='conv2_1_proj', 
            shape=(64, 32, 1, 1), pad=(0,0), stride=(1,1), no_bias=False)
    conv2_1 = conv2_1_3_conv + conv2_1_proj

    # stack up mCReLU layers
    n_curr_ch = 64
    conv2_2, n_curr_ch = mCReLU(data=conv2_1, group_name='conv2_2', 
            filters=(24, 24, 64), strides=((1,1),(1,1),(1,1)), use_global=is_test, n_curr_ch=n_curr_ch)
    conv2_3, n_curr_ch = mCReLU(data=conv2_2, group_name='conv2_3', 
            filters=(24, 24, 64), strides=((1,1),(1,1),(1,1)), use_global=is_test, n_curr_ch=n_curr_ch)
    conv3_1, n_curr_ch = mCReLU(data=conv2_3, group_name='conv3_1', 
            filters=(48, 48, 128), strides=((2,2),(1,1),(1,1)), use_global=is_test, n_curr_ch=n_curr_ch)
    conv3_2, n_curr_ch = mCReLU(data=conv3_1, group_name='conv3_2', 
            filters=(48, 48, 128), strides=((1,1),(1,1),(1,1)), use_global=is_test, n_curr_ch=n_curr_ch)
    conv3_3, n_curr_ch = mCReLU(data=conv3_2, group_name='conv3_3', 
            filters=(48, 48, 128), strides=((1,1),(1,1),(1,1)), use_global=is_test, n_curr_ch=n_curr_ch)
    conv3_4, n_curr_ch = mCReLU(data=conv3_3, group_name='conv3_4', 
            filters=(48, 48, 128), strides=((1,1),(1,1),(1,1)), use_global=is_test, n_curr_ch=n_curr_ch)
    nch_3_4 = n_curr_ch

    # stack up inception layers
    conv4_1, n_curr_ch = inception(data=conv3_4, group_name='conv4_1', 
            filter_1=64, filters_3=(48,128), filters_5=(24,48,48), filter_p=128, filter_out=256, 
            stride=(2,2), use_global=is_test, n_curr_ch=n_curr_ch)
    conv4_2, n_curr_ch = inception(data=conv4_1, group_name='conv4_2', 
            filter_1=64, filters_3=(64,128), filters_5=(24,48,48), filter_p=None, filter_out=256, 
            stride=(1,1), use_global=is_test, n_curr_ch=n_curr_ch)
    conv4_3, n_curr_ch = inception(data=conv4_2, group_name='conv4_3', 
            filter_1=64, filters_3=(64,128), filters_5=(24,48,48), filter_p=None, filter_out=256, 
            stride=(1,1), use_global=is_test, n_curr_ch=n_curr_ch)
    conv4_4, n_curr_ch = inception(data=conv4_3, group_name='conv4_4', 
            filter_1=64, filters_3=(64,128), filters_5=(24,48,48), filter_p=None, filter_out=256, 
            stride=(1,1), use_global=is_test, n_curr_ch=n_curr_ch)
    nch_4_4 = n_curr_ch

    conv5_1, n_curr_ch = inception(data=conv4_4, group_name='conv5_1', 
            filter_1=64, filters_3=(96,192), filters_5=(32,64,64), filter_p=128, filter_out=384, 
            stride=(2,2), use_global=is_test, n_curr_ch=n_curr_ch)
    conv5_2, n_curr_ch = inception(data=conv5_1, group_name='conv5_2', 
            filter_1=64, filters_3=(96,192), filters_5=(32,64,64), filter_p=None, filter_out=384, 
            stride=(1,1), use_global=is_test, n_curr_ch=n_curr_ch)
    conv5_3, n_curr_ch = inception(data=conv5_2, group_name='conv5_3', 
            filter_1=64, filters_3=(96,192), filters_5=(32,64,64), filter_p=None, filter_out=384, 
            stride=(1,1), use_global=is_test, n_curr_ch=n_curr_ch)
    conv5_4, n_curr_ch = inception(data=conv5_3, group_name='conv5_4', 
            filter_1=64, filters_3=(96,192), filters_5=(32,64,64), filter_p=None, filter_out=384, 
            stride=(1,1), use_global=is_test, n_curr_ch=n_curr_ch, final_bn=True)

    # final layers
    conv5_4_last_bn = mx.symbol.BatchNorm(name='conv5_4_last_bn', data=conv5_4, use_global_stats=is_test)
    conv5_4_last_relu = mx.symbol.Activation(name='conv5_4_last_relu', data=conv5_4_last_bn, act_type='relu')

    # hyperfeature
    downsample = mx.symbol.Pooling(name='downsample', data=conv3_4, pooling_convention='full', 
            kernel=(3,3), stride=(2,2), pool_type='max')
    upsample = mx.symbol.Deconvolution(name='upsample', data=conv5_4_last_relu, 
            num_filter=384, pad=(1,1), kernel=(4,4), stride=(2,2), num_group=384, no_bias=True)
    concat = mx.symbol.Concat(name='concat', 
            *[downsample, mx.symbol.Crop(conv4_4, downsample), mx.symbol.Crop(upsample, downsample)])
    convf_rpn = conv_twn(concat, name='convf_rpn', 
            shape=(128, nch_3_4+nch_4_4+384, 1, 1), pad=(0,0), stride=(1,1), no_bias=False)
    # convf_rpn = mx.symbol.Convolution(name='convf_rpn', data=concat, 
    #         num_filter=128, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
    reluf_rpn = mx.symbol.Activation(name='reluf_rpn', data=convf_rpn, act_type='relu')

    convf_2 = conv_twn(concat, name='convf_2', 
            shape=(384, nch_3_4+nch_4_4+384, 1, 1), pad=(0,0), stride=(1,1), no_bias=False)
    reluf_2 = mx.symbol.Activation(name='reluf_2', data=convf_2, act_type='relu')
    concat_convf = mx.symbol.Concat(name='concat_convf', *[reluf_rpn, reluf_2] )

    return reluf_rpn, concat_convf

def get_pvanet_twn_train(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS):
    data = mx.symbol.Variable(name="data")
    im_info = mx.symbol.Variable(name="im_info")
    gt_boxes = mx.symbol.Variable(name="gt_boxes")
    rpn_label = mx.symbol.Variable(name='label')
    rpn_bbox_target = mx.symbol.Variable(name='bbox_target')
    rpn_bbox_weight = mx.symbol.Variable(name='bbox_weight')

    # shared conv layers
    reluf_rpn, concat_convf = pvanet_preact(data, is_test=True)
    
    # RPN layers
    rpn_conv1 = conv_twn(reluf_rpn, name='rpn_conv1', 
            shape=(384, 128, 3, 3), pad=(1,1), stride=(1,1), no_bias=False)
    # rpn_conv1 = mx.symbol.Convolution(name='rpn_conv1', data=reluf_rpn, 
    #         num_filter=384, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    rpn_relu1 = mx.symbol.Activation(name='rpn_relu1', data=rpn_conv1, act_type='relu')
    rpn_cls_score = mx.symbol.Convolution(name='rpn_cls_score', data=rpn_relu1, 
            num_filter=2*num_anchors, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
    rpn_bbox_pred = mx.symbol.Convolution(name='rpn_bbox_pred', data=rpn_relu1, 
            num_filter=4*num_anchors, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)

    # prepare rpn data
    rpn_cls_score_reshape = mx.symbol.Reshape(
        data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")

    # classification
    rpn_cls_prob = mx.symbol.SoftmaxOutput(data=rpn_cls_score_reshape, label=rpn_label, multi_output=True,
            normalization='valid', use_ignore=True, ignore_label=-1, name="rpn_cls_prob")
    # bounding box regression
    rpn_bbox_loss_ = rpn_bbox_weight * \
            mx.symbol.smooth_l1(name='rpn_bbox_loss_', scalar=3.0, data=(rpn_bbox_pred - rpn_bbox_target))
    rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss', data=rpn_bbox_loss_, 
            grad_scale=1.0 / config.TRAIN.RPN_BATCH_SIZE)

    # ROI proposal
    rpn_cls_act = mx.symbol.SoftmaxActivation(
        data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_act")
    rpn_cls_act_reshape = mx.symbol.Reshape(
        data=rpn_cls_act, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_act_reshape')
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
        rois = mx.symbol.Custom(
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
    gt_boxes_reshape = mx.symbol.Reshape(data=gt_boxes, shape=(-1, 5), name='gt_boxes_reshape')
    group = mx.symbol.Custom(rois=rois, gt_boxes=gt_boxes_reshape, op_type='proposal_target',
                             num_classes=num_classes, batch_images=config.TRAIN.BATCH_IMAGES,
                             batch_rois=config.TRAIN.BATCH_ROIS, fg_fraction=config.TRAIN.FG_FRACTION)
    rois = group[0]
    label = group[1]
    bbox_target = group[2]
    bbox_weight = group[3]

    # Fast R-CNN
    roi_pool = mx.symbol.ROIPooling(name='roi_pool5', data=concat_convf, rois=rois, 
            pooled_size=(6, 6), spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)
    flat5 = mx.symbol.Flatten(name='flat5', data=roi_pool)
    fc6 = fc_twn(flat5, (4096, 6*6*512), name='fc6', no_bias=False)
    # fc6 = mx.symbol.FullyConnected(name='fc6', data=flat5, num_hidden=4096, no_bias=False)
    fc6_bn = mx.symbol.BatchNorm(name='fc6_bn', data=fc6, use_global_stats=True, fix_gamma=False)
    fc6_dropout = mx.symbol.Dropout(name='fc6_dropout', data=fc6_bn, p=0.25)
    fc6_relu = mx.symbol.Activation(name='fc6_relu', data=fc6_dropout, act_type='relu')
    fc7 = fc_twn(fc6_relu, (4096, 4096), name='fc7', no_bias=False)
    # fc7 = mx.symbol.FullyConnected(name='fc7', data=fc6_relu, num_hidden=4096, no_bias=False)
    fc7_bn = mx.symbol.BatchNorm(name='fc7_bn', data=fc7, use_global_stats=True, fix_gamma=False)
    fc7_dropout = mx.symbol.Dropout(name='fc7_dropout', data=fc7_bn, p=0.25)
    fc7_relu = mx.symbol.Activation(name='fc7_relu', data=fc7_dropout, act_type='relu')

    # classification
    cls_score = mx.symbol.FullyConnected(name='cls_score', data=fc7_relu, num_hidden=num_classes)
    cls_prob = mx.symbol.SoftmaxOutput(name='cls_prob', data=cls_score, label=label, normalization='batch')
    # bounding box regression
    bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=fc7_relu, num_hidden=num_classes*4)
    bbox_loss_ = bbox_weight * \
            mx.symbol.smooth_l1(name='bbox_loss_', scalar=1.0, data=(bbox_pred - bbox_target))
    bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / config.TRAIN.BATCH_ROIS)

    # reshape output
    label = mx.symbol.Reshape(name='label_reshape', data=label, shape=(config.TRAIN.BATCH_IMAGES, -1))
    cls_prob = mx.symbol.Reshape(name='cls_prob_reshape', data=cls_prob, 
            shape=(config.TRAIN.BATCH_IMAGES, -1, num_classes))
    bbox_loss = mx.symbol.Reshape(name='bbox_loss_reshape', data=bbox_loss, 
            shape=(config.TRAIN.BATCH_IMAGES, -1, 4 * num_classes))

    group = mx.symbol.Group([rpn_cls_prob, rpn_bbox_loss, cls_prob, bbox_loss, mx.symbol.BlockGrad(label)])
    return group

def get_pvanet_twn_test(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS):
    data = mx.symbol.Variable(name="data")
    im_info = mx.symbol.Variable(name="im_info")

    # shared conv layers
    reluf_rpn, concat_convf = pvanet_preact(data, is_test=True)

    # RPN layers
    rpn_conv1 = conv_twn(reluf_rpn, name='rpn_conv1', 
            shape=(384, 128, 3, 3), pad=(1,1), stride=(1,1), no_bias=False)
    rpn_relu1 = mx.symbol.Activation(name='rpn_relu1', data=rpn_conv1, act_type='relu')
    rpn_cls_score = mx.symbol.Convolution(name='rpn_cls_score', data=rpn_relu1, 
            num_filter=2*num_anchors, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
    rpn_bbox_pred = mx.symbol.Convolution(name='rpn_bbox_pred', data=rpn_relu1, 
            num_filter=4*num_anchors, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)

    # ROI Proposal
    rpn_cls_score_reshape = mx.symbol.Reshape(
        data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")
    rpn_cls_prob = mx.symbol.SoftmaxActivation(
        data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_prob")
    rpn_cls_prob_reshape = mx.symbol.Reshape(
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
        rois = mx.symbol.Custom(
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
    roi_pool = mx.symbol.ROIPooling(name='roi_pool5', data=concat_convf, rois=rois, 
            pooled_size=(6, 6), spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)
    flat5 = mx.symbol.Flatten(name='flat5', data=roi_pool)
    fc6 = fc_twn(flat5, (4096, 6*6*512), name='fc6', no_bias=False)
    # fc6 = mx.symbol.FullyConnected(name='fc6', data=flat5, num_hidden=4096, no_bias=False)
    fc6_bn = mx.symbol.BatchNorm(name='fc6_bn', data=fc6, use_global_stats=True, fix_gamma=False)
    fc6_relu = mx.symbol.Activation(name='fc6_relu', data=fc6_bn, act_type='relu')
    fc7 = fc_twn(fc6_relu, (4096, 4096), name='fc7', no_bias=False)
    # fc7 = mx.symbol.FullyConnected(name='fc7', data=fc6_relu, num_hidden=4096, no_bias=False)
    fc7_bn = mx.symbol.BatchNorm(name='fc7_bn', data=fc7, use_global_stats=True, fix_gamma=False)
    fc7_relu = mx.symbol.Activation(name='fc7_relu', data=fc7_bn, act_type='relu')

    # classification
    cls_score = mx.symbol.FullyConnected(name='cls_score', data=fc7_relu, num_hidden=num_classes)
    cls_prob = mx.symbol.SoftmaxActivation(name='cls_prob', data=cls_score)
    # bounding box regression
    bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=fc7_relu, num_hidden=num_classes*4)

    # reshape output
    cls_prob = mx.symbol.Reshape(name='cls_prob_reshape', data=cls_prob, 
            shape=(config.TEST.BATCH_IMAGES, -1, num_classes))
    bbox_pred = mx.symbol.Reshape(name='bbox_pred_reshape', data=bbox_pred, 
            shape=(config.TEST.BATCH_IMAGES, -1, 4 * num_classes))

    # group output
    group = mx.symbol.Group([rois, cls_prob, bbox_pred])
    return group
