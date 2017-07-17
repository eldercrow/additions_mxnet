import mxnet as mx
from numpy import power
from common import convolution, fullyconnected, batchnorm, multibox_layer_python


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


def pvanet_multibox(data, num_classes,
        patch_size=512, per_cls_reg=False, use_global_stats=True, no_bias=False, lr_mult=1.0):
    ''' pvanet 10.1 '''
    conv1 = conv_bn_relu(data, group_name='conv1',
            num_filter=12, kernel=(4,4), pad=(1,1), stride=(2,2), no_bias=no_bias,
            use_global_stats=use_global_stats, use_crelu=True, lr_mult=lr_mult)
    # conv2
    conv2 = mcrelu(conv1, prefix_group='conv2',
            filters=(12, 18, 36), no_bias=no_bias, use_global_stats=use_global_stats, lr_mult=lr_mult)
    # conv3
    conv3 = mcrelu(conv2, prefix_group='conv3',
            filters=(18, 36, 72), no_bias=no_bias, use_global_stats=use_global_stats, lr_mult=lr_mult)
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

    # upsampled hyperfeature
    projc = conv_bn_relu(concat, group_name='projc', 
            num_filter=128, kernel=(1,1), pad=(0,0), stride=(1,1), no_bias=True,
            use_global_stats=use_global_stats, lr_mult=lr_mult)
    upc = conv_bn_relu(projc, group_name='upc', 
            num_filter=128, kernel=(3,3), pad=(1,1), stride=(1,1), no_bias=True,
            use_global_stats=use_global_stats, lr_mult=lr_mult)
    upc = mx.sym.UpSampling(upc, scale=2, sample_type='bilinear', num_filter=128, num_args=2)
    res3  = mx.sym.concat(conv3, upc)
    res3 = conv_bn_relu(res3, group_name='hyper3',
            num_filter=256, kernel=(1,1), pad=(0,0), stride=(1,1), no_bias=True,
            use_global_stats=use_global_stats, lr_mult=lr_mult)
    from_layers = [res3]

    sz_ratio = power(2.0, 0.25)
    rfs = [40, 80, 160, 320, 640]
    strides = [r / 5 for r in rfs]

    convf = concat
    for i, r in enumerate(rfs[1:], 1):
        if i > 1:
            convf = mx.sym.Pooling(convf, kernel=(2,2), stride=(2,2), pool_type='max')
        projf = conv_bn_relu(convf, group_name='projf_{}'.format(r),
                num_filter=128, kernel=(1,1), pad=(0,0), stride=(1,1), no_bias=True,
                use_global_stats=use_global_stats, lr_mult=lr_mult)
        convf = conv_bn_relu(projf, group_name='convf_{}'.format(r),
                num_filter=256, kernel=(3,3), pad=(1,1), stride=(1,1), no_bias=True,
                use_global_stats=use_global_stats, lr_mult=lr_mult)
        from_layers.append(convf)

    ratios = [[1.0, 2.0/3.0, 3.0/2.0, 4.0/9.0, 9.0/4.0]] * len(from_layers)
    sizes = [[r / sz_ratio, r * sz_ratio] for r in rfs[:-1]]
    sizes.append([rfs[-1] / sz_ratio,])

    preds, anchors = multibox_layer_python(from_layers, num_classes,
            sizes=sizes, ratios=ratios, strides=strides, per_cls_reg=per_cls_reg, clip=False)
    return preds, anchors
