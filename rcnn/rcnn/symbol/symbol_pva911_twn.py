import mxnet as mx
import proposal
import proposal_target
from rcnn.config import config
from ternarize_ch import *


def conv_twn(data, num_filter, nch, name=None, kernel=(1,1), pad=(0,0), stride=(1,1), no_bias=False):
    ''' ternary weight convolution '''
    shape = (num_filter, nch, kernel[0], kernel[1])
    conv_weight = mx.sym.var(name=name+'_weight', shape=shape, attr={'__wd_mult__': '0.0'}, dtype='float32')
    weight = mx.sym.Custom(conv_weight, op_type='ternarize_ch', filterwise='channel')
    conv = mx.sym.Convolution(data=data, weight=weight, name=name, num_filter=num_filter, 
            kernel=kernel, pad=pad, stride=stride, no_bias=no_bias)
    return conv


def fc_twn(data, num_hidden, nch, name=None, no_bias=False):
    ''' ternary weight fc '''
    fc_weight = mx.sym.var(name=name+'_weight', shape=(num_hidden, nch), 
            attr={'__wd_mult__': '0.0'}, dtype='float32')
    # weight, alpha = mx.sym.Ternarize(fc_weight)
    # weight = mx.sym.broadcast_mul(weight, alpha)
    weight = mx.sym.Custom(fc_weight, op_type='ternarize_ch', filterwise='channel', th_ratio=0.7)
    fc = mx.sym.FullyConnected(data=data, weight=weight, name=name, num_hidden=num_hidden, no_bias=no_bias)
    return fc


def conv_bn_relu(data,
                 group_name,
                 num_filter,
                 kernel,
                 pad,
                 stride,
                 use_global=True):
    """ used in mCReLU """
    bn_name = group_name + '/bn'
    relu_name = group_name + '/relu'
    conv_name = group_name + '/conv'

    conv = mx.sym.Convolution(
        name=conv_name,
        data=data,
        num_filter=num_filter,
        pad=pad,
        kernel=kernel,
        stride=stride,
        no_bias=True)
    bn = mx.sym.BatchNorm(
        name=bn_name,
        data=conv,
        use_global_stats=use_global,
        fix_gamma=False,
        eps=1e-05)
    relu = mx.sym.Activation(name=relu_name, data=bn, act_type='relu')
    return relu


def bn_relu_conv(data,
                 group_name,
                 num_filter,
                 kernel,
                 pad,
                 stride,
                 use_global=True,
                 use_crelu=False,
                 get_syms=False):
    """ used in mCReLU """
    bn_name = group_name + '/bn'
    relu_name = group_name + '/relu'
    conv_name = group_name + '/conv'
    concat_name = group_name + '/concat'

    if use_crelu:
        data = mx.sym.Concat(name=concat_name, *[data, -data])
    bn = mx.sym.BatchNorm(
        name=bn_name,
        data=data,
        use_global_stats=use_global,
        fix_gamma=False,
        eps=1e-05)
    relu = mx.sym.Activation(name=relu_name, data=bn, act_type='relu')
    conv = mx.sym.Convolution(
        name=conv_name,
        data=relu,
        num_filter=num_filter,
        pad=pad,
        kernel=kernel,
        stride=stride,
        no_bias=False,
        cudnn_off=False)
    if get_syms:
        syms = {'bn': bn, 'relu': relu, 'conv': conv}
        return conv, syms
    else:
        return conv


def mCReLU(data, group_name, filters, strides, use_global, n_curr_ch):
    """
    """
    kernels = ((1, 1), (3, 3), (1, 1))
    pads = ((0, 0), (1, 1), (0, 0))
    crelu = (False, False, True)

    # syms = {}
    ssyms = []

    conv = data
    for i in range(3):
        conv, s = bn_relu_conv(
            data=conv,
            group_name=group_name + '/{}'.format(i + 1),
            num_filter=filters[i],
            pad=pads[i],
            kernel=kernels[i],
            stride=strides[i],
            use_global=use_global,
            use_crelu=crelu[i],
            get_syms=True)
        # syms['conv{}'.format(i+1)] = conv
        ssyms.append(s)

    ss = 1
    for s in strides:
        ss *= s[0]
    need_proj = (n_curr_ch != filters[2]) or (ss != 1)
    if need_proj:
        proj = mx.sym.Convolution(
            data=ssyms[0]['relu'],
            name=group_name + '/proj',
            num_filter=filters[2],
            pad=(0, 0),
            kernel=(1, 1),
            stride=(ss, ss),
            cudnn_off=False)
        res = conv + proj
        return res, filters[2], proj
        # syms['proj'] = proj
    else:
        res = conv + data
        return res, filters[2]
    # return res, filters[2]# , syms


    # final_bn is to handle stupid redundancy in the original model
def inception(data,
              group_name,
              filter_0,
              filters_1,
              filters_2,
              filter_p,
              filter_out,
              stride,
              use_global,
              n_curr_ch,
              final_bn=False):
    """
    """
    incep_name = group_name + '/incep'

    group_name_0 = incep_name + '/0'
    group_name_1 = incep_name + '/1'
    group_name_2 = incep_name + '/2'

    incep_bn = mx.sym.BatchNorm(
        name=incep_name + '/bn',
        data=data,
        use_global_stats=use_global,
        fix_gamma=False,
        eps=1e-05)
    incep_relu = mx.sym.Activation(
        name=incep_name + '/relu', data=incep_bn, act_type='relu')

    incep_0 = conv_bn_relu(
        data=incep_relu,
        group_name=group_name_0,
        num_filter=filter_0,
        kernel=(1, 1),
        pad=(0, 0),
        stride=stride,
        use_global=use_global)

    incep_1_reduce = conv_bn_relu(
        data=incep_relu,
        group_name=group_name_1 + '_reduce',
        num_filter=filters_1[0],
        kernel=(1, 1),
        pad=(0, 0),
        stride=stride,
        use_global=use_global)
    incep_1_0 = conv_bn_relu(
        data=incep_1_reduce,
        group_name=group_name_1 + '_0',
        num_filter=filters_1[1],
        kernel=(3, 3),
        pad=(1, 1),
        stride=(1, 1),
        use_global=use_global)

    incep_2_reduce = conv_bn_relu(
        data=incep_relu,
        group_name=group_name_2 + '_reduce',
        num_filter=filters_2[0],
        kernel=(1, 1),
        pad=(0, 0),
        stride=stride,
        use_global=use_global)
    incep_2_0 = conv_bn_relu(
        data=incep_2_reduce,
        group_name=group_name_2 + '_0',
        num_filter=filters_2[1],
        kernel=(3, 3),
        pad=(1, 1),
        stride=(1, 1),
        use_global=use_global)
    incep_2_1 = conv_bn_relu(
        data=incep_2_0,
        group_name=group_name_2 + '_1',
        num_filter=filters_2[2],
        kernel=(3, 3),
        pad=(1, 1),
        stride=(1, 1),
        use_global=use_global)

    incep_layers = [incep_0, incep_1_0, incep_2_1]
    # incep_layers = [incep_0, incep_2_1]

    if filter_p is not None:
        incep_p_pool = mx.sym.Pooling(
            name=incep_name + '/pool',
            data=incep_relu,
            pooling_convention='full',
            pad=(0, 0),
            kernel=(3, 3),
            stride=(2, 2),
            pool_type='max')
        incep_p_proj = conv_bn_relu(
            data=incep_p_pool,
            group_name=incep_name + '/poolproj',
            num_filter=filter_p,
            kernel=(1, 1),
            pad=(0, 0),
            stride=(1, 1),
            use_global=use_global)
        incep_layers.append(incep_p_proj)

    incep = mx.sym.concat(*incep_layers, name=incep_name)
    # out_conv = mx.sym.Convolution(name=group_name.replace('_incep', '_out_conv'), data=incep,
    #         num_filter=filter_out, kernel=(1,1), stride=(1,1), pad=(0,0))

    # final_bn is to handle stupid redundancy in the original model
    if final_bn:
        out_conv = mx.sym.Convolution(
            name=group_name + '/out/conv',
            data=incep,
            num_filter=filter_out,
            kernel=(1, 1),
            stride=(1, 1),
            pad=(0, 0),
            no_bias=True)
        out_conv = mx.sym.BatchNorm(
            name=group_name + '/out/bn',
            data=out_conv,
            use_global_stats=use_global,
            fix_gamma=False,
            eps=1e-05)
    else:
        out_conv = mx.sym.Convolution(
            name=group_name + '/out/conv',
            data=incep,
            num_filter=filter_out,
            kernel=(1, 1),
            stride=(1, 1),
            pad=(0, 0))

    if n_curr_ch != filter_out or stride[0] > 1:
        out_proj = mx.sym.Convolution(
            name=group_name + '/proj',
            data=data,
            num_filter=filter_out,
            kernel=(1, 1),
            stride=stride,
            pad=(0, 0))
        return out_conv + out_proj, filter_out, out_proj
    else:
        return out_conv + data, filter_out


def pvanet_preact(data, is_test):
    """ PVANet 9.0 """
    conv1_1_conv = mx.sym.Convolution(
        name='conv1_1/conv',
        data=data,
        num_filter=16,
        pad=(3, 3),
        kernel=(7, 7),
        stride=(2, 2),
        no_bias=True)
    conv1_1_concat = mx.sym.Concat(
        name='conv1_1/concat', *[conv1_1_conv, -conv1_1_conv])
    conv1_1_bn = mx.sym.BatchNorm(
        name='conv1_1/bn',
        data=conv1_1_concat,
        use_global_stats=is_test,
        fix_gamma=False,
        eps=1e-05)
    conv1_1_relu = mx.sym.Activation(
        name='conv1_1/relu', data=conv1_1_bn, act_type='relu')
    pool1 = mx.sym.Pooling(
        name='pool1',
        data=conv1_1_relu,
        pooling_convention='full',
        pad=(0, 0),
        kernel=(3, 3),
        stride=(2, 2),
        pool_type='max')

    # no pre bn-scale-relu for 2_1_1
    conv2_1_1_conv = mx.sym.Convolution(
        name='conv2_1/1/conv',
        data=pool1,
        num_filter=24,
        pad=(0, 0),
        kernel=(1, 1),
        stride=(1, 1),
        no_bias=False)
    conv2_1_2_conv = bn_relu_conv(
        data=conv2_1_1_conv,
        group_name='conv2_1/2',
        num_filter=24,
        kernel=(3, 3),
        pad=(1, 1),
        stride=(1, 1),
        use_global=is_test)
    conv2_1_3_conv = bn_relu_conv(
        data=conv2_1_2_conv,
        group_name='conv2_1/3',
        num_filter=64,
        kernel=(1, 1),
        pad=(0, 0),
        stride=(1, 1),
        use_global=is_test,
        use_crelu=True)
    conv2_1_proj = mx.sym.Convolution(
        name='conv2_1/proj',
        data=pool1,
        num_filter=64,
        pad=(0, 0),
        kernel=(1, 1),
        stride=(1, 1),
        no_bias=False)
    conv2_1 = conv2_1_3_conv + conv2_1_proj

    # stack up mCReLU layers
    n_curr_ch = 64
    conv2_2, n_curr_ch = mCReLU(
        data=conv2_1,
        group_name='conv2_2',
        filters=(24, 24, 64),
        strides=((1, 1), (1, 1), (1, 1)),
        use_global=is_test,
        n_curr_ch=n_curr_ch)
    conv2_3, n_curr_ch = mCReLU(
        data=conv2_2,
        group_name='conv2_3',
        filters=(24, 24, 64),
        strides=((1, 1), (1, 1), (1, 1)),
        use_global=is_test,
        n_curr_ch=n_curr_ch)
    conv2_3_ror = conv2_1_proj + conv2_3
    conv3_1, n_curr_ch, conv3_1_proj = mCReLU(
        data=conv2_3_ror,
        group_name='conv3_1',
        filters=(48, 48, 128),
        strides=((2, 2), (1, 1), (1, 1)),
        use_global=is_test,
        n_curr_ch=n_curr_ch)
    conv3_2, n_curr_ch = mCReLU(
        data=conv3_1,
        group_name='conv3_2',
        filters=(48, 48, 128),
        strides=((1, 1), (1, 1), (1, 1)),
        use_global=is_test,
        n_curr_ch=n_curr_ch)
    conv3_3, n_curr_ch = mCReLU(
        data=conv3_2,
        group_name='conv3_3',
        filters=(48, 48, 128),
        strides=((1, 1), (1, 1), (1, 1)),
        use_global=is_test,
        n_curr_ch=n_curr_ch)
    conv3_4, n_curr_ch = mCReLU(
        data=conv3_3,
        group_name='conv3_4',
        filters=(48, 48, 128),
        strides=((1, 1), (1, 1), (1, 1)),
        use_global=is_test,
        n_curr_ch=n_curr_ch)
    conv3_4_ror = conv3_4 + conv3_1_proj

    # stack up inception layers
    conv4_1, n_curr_ch, conv4_1_proj = inception(
        data=conv3_4_ror, group_name='conv4_1',
        filter_0=64, filters_1=(48, 128), filters_2=(24, 48, 48), filter_p=128, filter_out=256,
        stride=(2, 2), use_global=is_test, n_curr_ch=n_curr_ch)
    conv4_2, n_curr_ch = inception(
        data=conv4_1,
        group_name='conv4_2',
        filter_0=64,
        filters_1=(64, 128),
        filters_2=(24, 48, 48),
        filter_p=None,
        filter_out=256,
        stride=(1, 1),
        use_global=is_test,
        n_curr_ch=n_curr_ch)
    conv4_3, n_curr_ch = inception(
        data=conv4_2,
        group_name='conv4_3',
        filter_0=64,
        filters_1=(64, 128),
        filters_2=(24, 48, 48),
        filter_p=None,
        filter_out=256,
        stride=(1, 1),
        use_global=is_test,
        n_curr_ch=n_curr_ch)
    conv4_4, n_curr_ch = inception(
        data=conv4_3,
        group_name='conv4_4',
        filter_0=64,
        filters_1=(64, 128),
        filters_2=(24, 48, 48),
        filter_p=None,
        filter_out=256,
        stride=(1, 1),
        use_global=is_test,
        n_curr_ch=n_curr_ch)
    conv4_4_ror = conv4_4 + conv4_1_proj
    conv5_1, n_curr_ch, conv5_1_proj = inception(
        data=conv4_4_ror,
        group_name='conv5_1',
        filter_0=64,
        filters_1=(96, 192),
        filters_2=(32, 64, 64),
        filter_p=128,
        filter_out=384,
        stride=(2, 2),
        use_global=is_test,
        n_curr_ch=n_curr_ch)
    conv5_2, n_curr_ch = inception(
        data=conv5_1,
        group_name='conv5_2',
        filter_0=64,
        filters_1=(96, 192),
        filters_2=(32, 64, 64),
        filter_p=None,
        filter_out=384,
        stride=(1, 1),
        use_global=is_test,
        n_curr_ch=n_curr_ch)
    conv5_3, n_curr_ch = inception(
        data=conv5_2,
        group_name='conv5_3',
        filter_0=64,
        filters_1=(96, 192),
        filters_2=(32, 64, 64),
        filter_p=None,
        filter_out=384,
        stride=(1, 1),
        use_global=is_test,
        n_curr_ch=n_curr_ch)
    conv5_4, n_curr_ch = inception(
        data=conv5_3,
        group_name='conv5_4',
        filter_0=64,
        filters_1=(96, 192),
        filters_2=(32, 64, 64),
        filter_p=None,
        filter_out=384,
        stride=(1, 1),
        use_global=is_test,
        n_curr_ch=n_curr_ch,
        final_bn=True)
    conv5_4_ror = conv5_4 + conv5_1_proj

    conv2_1_long_proj = mx.sym.Convolution(
        pool1,
        name='conv2_1/long/proj',
        num_filter=384,
        kernel=(1, 1),
        stride=(8, 8))
    conv5_4_long_ror = conv5_4_ror + conv2_1_long_proj

    # final layers
    conv5_4_last_bn = mx.sym.BatchNorm(
        conv5_4_long_ror,
        name='conv5_4/last_bn',
        use_global_stats=is_test,
        eps=1e-05,
        fix_gamma=False)
    conv5_4_last_relu = mx.sym.Activation(
        conv5_4_last_bn, name='conv5_4/last_relu', act_type='relu')

    # hyperfeature
    downsample = mx.sym.Pooling(
        name='downsample',
        data=conv3_4_ror,
        pooling_convention='full',
        kernel=(3, 3),
        stride=(2, 2),
        pool_type='max')
    upsample = mx.sym.UpSampling(
        name='upsample',
        data=conv5_4_last_relu,
        scale=2,
        sample_type='bilinear',
        num_filter=384,
        num_args=2)
    # upsample = mx.sym.Deconvolution(name='upsample', data=conv5_4_last_relu,
    #         num_filter=384, pad=(1,1), kernel=(4,4), stride=(2,2), num_group=384, no_bias=True)
    concat = mx.sym.Concat(name='concat', *[downsample, conv4_4_ror, upsample])
    convf_rpn = mx.sym.Convolution(
        name='convf_rpn',
        data=concat,
        num_filter=128,
        pad=(0, 0),
        kernel=(1, 1),
        stride=(1, 1))
    reluf_rpn = mx.sym.Activation(
        name='reluf_rpn', data=convf_rpn, act_type='relu')

    convf_2 = mx.sym.Convolution(
        name='convf_2',
        data=concat,
        num_filter=384,
        pad=(0, 0),
        kernel=(1, 1),
        stride=(1, 1))
    reluf_2 = mx.sym.Activation(name='reluf_2', data=convf_2, act_type='relu')
    concat_convf = mx.sym.Concat(name='concat_convf', *[reluf_rpn, reluf_2])

    return reluf_rpn, concat_convf


def get_pvanet_twn_train(num_classes=config.NUM_CLASSES,
                     num_anchors=config.NUM_ANCHORS):
    data = mx.sym.Variable(name="data")
    im_info = mx.sym.Variable(name="im_info")
    gt_boxes = mx.sym.Variable(name="gt_boxes")
    rpn_label = mx.sym.Variable(name='label')
    rpn_bbox_target = mx.sym.Variable(name='bbox_target')
    rpn_bbox_weight = mx.sym.Variable(name='bbox_weight')

    # shared conv layers
    reluf_rpn, concat_convf = pvanet_preact(data, is_test=True)

    # RPN layers
    rpn_conv1 = mx.sym.Convolution(
        name='rpn_conv1',
        data=reluf_rpn,
        num_filter=384,
        pad=(1, 1),
        kernel=(3, 3),
        stride=(1, 1))
    rpn_relu1 = mx.sym.Activation(
        name='rpn_relu1', data=rpn_conv1, act_type='relu')
    rpn_cls_score = mx.sym.Convolution(
        name='rpn_cls_score',
        data=rpn_relu1,
        num_filter=2 * num_anchors,
        pad=(0, 0),
        kernel=(1, 1),
        stride=(1, 1),
        no_bias=False)
    rpn_bbox_pred = mx.sym.Convolution(
        name='rpn_bbox_pred',
        data=rpn_relu1,
        num_filter=4 * num_anchors,
        pad=(0, 0),
        kernel=(1, 1),
        stride=(1, 1),
        no_bias=False)

    # prepare rpn data
    rpn_cls_score_reshape = mx.sym.Reshape(
        data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")

    # classification
    rpn_cls_prob = mx.sym.SoftmaxOutput(
        data=rpn_cls_score_reshape,
        label=rpn_label,
        multi_output=True,
        normalization='valid',
        use_ignore=True,
        ignore_label=-1,
        name="rpn_cls_prob")
    # bounding box regression
    rpn_bbox_loss_ = rpn_bbox_weight * \
            mx.sym.smooth_l1(name='rpn_bbox_loss_', scalar=3.0, data=(rpn_bbox_pred - rpn_bbox_target))
    rpn_bbox_loss = mx.sym.MakeLoss(
        name='rpn_bbox_loss',
        data=rpn_bbox_loss_,
        grad_scale=1.0 / config.TRAIN.RPN_BATCH_SIZE)

    # ROI proposal
    rpn_cls_act = mx.sym.SoftmaxActivation(
        data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_act")
    rpn_cls_act_reshape = mx.sym.Reshape(
        data=rpn_cls_act,
        shape=(0, 2 * num_anchors, -1, 0),
        name='rpn_cls_act_reshape')
    if config.TRAIN.CXX_PROPOSAL:
        rois = mx.contrib.symbol.Proposal(
            cls_prob=rpn_cls_act_reshape,
            bbox_pred=rpn_bbox_pred,
            im_info=im_info,
            name='rois',
            feature_stride=config.RPN_FEAT_STRIDE,
            scales=tuple(config.ANCHOR_SCALES),
            ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=config.TRAIN.RPN_PRE_NMS_TOP_N,
            rpn_post_nms_top_n=config.TRAIN.RPN_POST_NMS_TOP_N,
            threshold=config.TRAIN.RPN_NMS_THRESH,
            rpn_min_size=config.TRAIN.RPN_MIN_SIZE)
    else:
        rois = mx.sym.Custom(
            cls_prob=rpn_cls_act_reshape,
            bbox_pred=rpn_bbox_pred,
            im_info=im_info,
            name='rois',
            op_type='proposal',
            feat_stride=config.RPN_FEAT_STRIDE,
            scales=tuple(config.ANCHOR_SCALES),
            ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=config.TRAIN.RPN_PRE_NMS_TOP_N,
            rpn_post_nms_top_n=config.TRAIN.RPN_POST_NMS_TOP_N,
            threshold=config.TRAIN.RPN_NMS_THRESH,
            rpn_min_size=config.TRAIN.RPN_MIN_SIZE)

    # ROI proposal target
    gt_boxes_reshape = mx.sym.Reshape(
        data=gt_boxes, shape=(-1, 5), name='gt_boxes_reshape')
    group = mx.sym.Custom(
        rois=rois,
        gt_boxes=gt_boxes_reshape,
        op_type='proposal_target',
        num_classes=num_classes,
        batch_images=config.TRAIN.BATCH_IMAGES,
        batch_rois=config.TRAIN.BATCH_ROIS,
        fg_fraction=config.TRAIN.FG_FRACTION)
    rois = group[0]
    label = group[1]
    bbox_target = group[2]
    bbox_weight = group[3]

    # Fast R-CNN
    roi_pool = mx.sym.ROIPooling(
        name='roi_pool5',
        data=concat_convf,
        rois=rois,
        pooled_size=(6, 6),
        spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)
    flat5 = mx.sym.Flatten(name='flat5', data=roi_pool)

    # fc6 = mx.sym.FullyConnected(
    #     name='fc6', data=flat5, num_hidden=4096, no_bias=False)
    fc6 = fc_twn(
        name='fc6', data=flat5, num_hidden=4096, nch=6*6*512, no_bias=False)
    fc6_bn = mx.sym.BatchNorm(
        name='fc6/bn',
        data=fc6,
        use_global_stats=True,
        fix_gamma=False,
        eps=1e-05)
    fc6_relu = mx.sym.Activation(name='fc6/relu', data=fc6_bn, act_type='relu')

    # fc7 = mx.sym.FullyConnected(
    #     name='fc7', data=fc6_relu, num_hidden=4096, no_bias=False)
    fc7 = fc_twn(
        name='fc7', data=fc6_relu, num_hidden=4096, nch=4096, no_bias=False)
    fc7_bn = mx.sym.BatchNorm(
        name='fc7/bn',
        data=fc7,
        use_global_stats=True,
        fix_gamma=False,
        eps=1e-05)
    fc7_relu = mx.sym.Activation(name='fc7/relu', data=fc7_bn, act_type='relu')

    # classification
    cls_score = mx.sym.FullyConnected(
        name='cls_score', data=fc7_relu, num_hidden=num_classes)
    cls_prob = mx.sym.SoftmaxOutput(
        name='cls_prob', data=cls_score, label=label, normalization='batch')
    # bounding box regression
    bbox_pred = mx.sym.FullyConnected(
        name='bbox_pred', data=fc7_relu, num_hidden=num_classes * 4)
    bbox_loss_ = bbox_weight * \
            mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0, data=(bbox_pred - bbox_target))
    bbox_loss = mx.sym.MakeLoss(
        name='bbox_loss',
        data=bbox_loss_,
        grad_scale=1.0 / config.TRAIN.BATCH_ROIS)

    # reshape output
    label = mx.sym.Reshape(
        name='label_reshape',
        data=label,
        shape=(config.TRAIN.BATCH_IMAGES, -1))
    cls_prob = mx.sym.Reshape(
        name='cls_prob_reshape',
        data=cls_prob,
        shape=(config.TRAIN.BATCH_IMAGES, -1, num_classes))
    bbox_loss = mx.sym.Reshape(
        name='bbox_loss_reshape',
        data=bbox_loss,
        shape=(config.TRAIN.BATCH_IMAGES, -1, 4 * num_classes))

    group = mx.sym.Group([
        rpn_cls_prob, rpn_bbox_loss, cls_prob, bbox_loss,
        mx.sym.BlockGrad(label)
    ])
    return group


def get_pvanet_twn_test(num_classes=config.NUM_CLASSES,
                    num_anchors=config.NUM_ANCHORS):
    data = mx.sym.Variable(name="data")
    im_info = mx.sym.Variable(name="im_info", init=mx.init.Zero())

    # shared conv layers
    reluf_rpn, concat_convf = pvanet_preact(data, is_test=True)

    # RPN layers
    rpn_conv1 = mx.sym.Convolution(
        reluf_rpn,
        name='rpn_conv1',
        num_filter=384,
        pad=(1, 1),
        kernel=(3, 3),
        stride=(1, 1),
        no_bias=False)
    rpn_relu1 = mx.sym.Activation(rpn_conv1, name='rpn_relu1', act_type='relu')
    rpn_cls_score = mx.sym.Convolution(
        rpn_relu1,
        name='rpn_cls_score',
        num_filter=2 * num_anchors,
        pad=(0, 0),
        kernel=(1, 1),
        stride=(1, 1),
        no_bias=False)
    rpn_bbox_pred = mx.sym.Convolution(
        rpn_relu1,
        name='rpn_bbox_pred',
        num_filter=4 * num_anchors,
        pad=(0, 0),
        kernel=(1, 1),
        stride=(1, 1),
        no_bias=False)

    # ROI Proposal
    rpn_cls_score_reshape = mx.sym.Reshape(
        data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")
    rpn_cls_prob = mx.sym.SoftmaxActivation(
        data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_prob")
    rpn_cls_prob_reshape = mx.sym.Reshape(
        data=rpn_cls_prob,
        shape=(0, 2 * num_anchors, -1, 0),
        name='rpn_cls_prob_reshape')
    if config.TEST.CXX_PROPOSAL:
        rois = mx.contrib.symbol.Proposal(
            cls_prob=rpn_cls_prob_reshape,
            bbox_pred=rpn_bbox_pred,
            im_info=im_info,
            name='rois',
            feature_stride=config.RPN_FEAT_STRIDE,
            scales=tuple(config.ANCHOR_SCALES),
            ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=config.TEST.RPN_PRE_NMS_TOP_N,
            rpn_post_nms_top_n=config.TEST.RPN_POST_NMS_TOP_N,
            threshold=config.TEST.RPN_NMS_THRESH,
            rpn_min_size=config.TEST.RPN_MIN_SIZE)
    else:
        rois = mx.sym.Custom(
            cls_prob=rpn_cls_prob_reshape,
            bbox_pred=rpn_bbox_pred,
            im_info=im_info,
            name='rois',
            op_type='proposal',
            feat_stride=config.RPN_FEAT_STRIDE,
            scales=tuple(config.ANCHOR_SCALES),
            ratios=tuple(config.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=config.TEST.RPN_PRE_NMS_TOP_N,
            rpn_post_nms_top_n=config.TEST.RPN_POST_NMS_TOP_N,
            threshold=config.TEST.RPN_NMS_THRESH,
            rpn_min_size=config.TEST.RPN_MIN_SIZE)

    # Fast R-CNN
    roi_pool = mx.sym.ROIPooling(
        name='roi_pool5',
        data=concat_convf,
        rois=rois,
        pooled_size=(6, 6),
        spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)
    flat5 = mx.sym.Flatten(name='flat5', data=roi_pool)

    fc6 = mx.sym.FullyConnected(
        name='fc6', data=flat5, num_hidden=4096, no_bias=False)
    fc6_bn = mx.sym.BatchNorm(
        name='fc6/bn',
        data=fc6,
        use_global_stats=True,
        fix_gamma=False,
        eps=1e-05)
    fc6_relu = mx.sym.Activation(name='fc6/relu', data=fc6_bn, act_type='relu')

    fc7 = mx.sym.FullyConnected(
        name='fc7', data=fc6_relu, num_hidden=4096, no_bias=False)
    fc7_bn = mx.sym.BatchNorm(
        name='fc7/bn',
        data=fc7,
        use_global_stats=True,
        fix_gamma=False,
        eps=1e-05)
    fc7_relu = mx.sym.Activation(name='fc7/relu', data=fc7_bn, act_type='relu')

    # classification
    cls_score = mx.sym.FullyConnected(
        name='cls_score', data=fc7_relu, num_hidden=num_classes)
    cls_prob = mx.sym.SoftmaxActivation(name='cls_prob', data=cls_score)
    # bounding box regression
    bbox_pred = mx.sym.FullyConnected(
        name='bbox_pred', data=fc7_relu, num_hidden=num_classes * 4)

    # reshape output
    cls_prob = mx.sym.Reshape(
        name='cls_prob_reshape',
        data=cls_prob,
        shape=(config.TEST.BATCH_IMAGES, -1, num_classes))
    bbox_pred = mx.sym.Reshape(
        name='bbox_pred_reshape',
        data=bbox_pred,
        shape=(config.TEST.BATCH_IMAGES, -1, 4 * num_classes))

    # group output
    group = mx.sym.Group([rois, cls_prob, bbox_pred])
    return group
