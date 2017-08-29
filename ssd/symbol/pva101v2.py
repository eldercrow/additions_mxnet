import mxnet as mx


def conv_bn_relu(data, group_name,
        num_filter, kernel, pad, stride=(1, 1), dilate=(1, 1), no_bias=True,
        use_global_stats=True, use_crelu=False,
        get_syms=False):
    ''' use in mCReLU '''
    conv_name = group_name + ''
    concat_name = group_name + '/concat'
    bn_name = group_name + '/bn'
    relu_name = group_name + '/relu'

    syms = {}
    conv = mx.sym.Convolution(data, name=conv_name,
            num_filter=num_filter, kernel=kernel, pad=pad, stride=stride, dilate=dilate, no_bias=no_bias)
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


def subpixel_upsample(data, ch, c, r):
    if r == 1 and c == 1:
        return data
    X = mx.sym.reshape(data=data, shape=(-3, 0, 0))  # (bsize*ch*r*c, a, b)
    X = mx.sym.reshape(
        data=X, shape=(-4, -1, r * c, 0, 0))  # (bsize*ch, r*c, a, b)
    X = mx.sym.transpose(data=X, axes=(0, 3, 2, 1))  # (bsize*ch, b, a, r*c)
    X = mx.sym.reshape(data=X, shape=(0, 0, -1, c))  # (bsize*ch, b, a*r, c)
    X = mx.sym.transpose(data=X, axes=(0, 2, 1, 3))  # (bsize*ch, a*r, b, c)
    X = mx.sym.reshape(
        data=X, shape=(-4, -1, ch, 0, -3))  # (bsize, ch, a*r, b*c)
    return X


def upsample_feature(data, name, scale,
                     num_filter_proj=0, num_filter_upsample=0,
                     use_global_stats=False):
    ''' use subpixel_upsample to upsample a given layer '''
    if num_filter_proj > 0:
        proj = conv_bn_relu(data, name+'proj/',
                num_filter=num_filter_proj, kernel=(1, 1), pad=(0, 0),
                use_global_stats=use_global_stats)
    else:
        proj = data
    nf = num_filter_upsample * scale * scale
    bn = conv_bn_relu(proj, name+'conv/',
            num_filter=nf, kernel=(3, 3), pad=(1, 1),
            use_global_stats=use_global_stats)
    return subpixel_upsample(bn, num_filter_upsample, scale, scale)


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
            num_filter=192, stride=(1,1), no_bias=no_bias, use_global_stats=use_global_stats)
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
            num_filter=256, stride=(1,1), no_bias=no_bias, use_global_stats=use_global_stats)

    hyper_group = []

    # hyper0
    up32 = upsample_feature(inc3e, name='up32', scale=2,
            num_filter_proj=64, num_filter_upsample=128, use_global_stats=use_global_stats)
    up42 = upsample_feature(inc4e, name='up42', scale=4,
            num_filter_proj=128, num_filter_upsample=64, use_global_stats=use_global_stats)
    hyper0 = mx.sym.concat(conv3, up32, up42)
    hyper0_0 = conv_bn_relu(hyper0, 'hyper0_0', num_filter=192,
            kernel=(1, 1), pad=(0, 0), use_global_stats=use_global_stats)
    hyper0_1 = conv_bn_relu(hyper0, 'hyper0_1', num_filter=192,
            kernel=(1, 1), pad=(0, 0), use_global_stats=use_global_stats)
    hyper_group.append(mx.sym.concat(hyper0_0, hyper0_1))

    # hyper1
    up43 = upsample_feature(inc4e, name='up43', scale=2,
            num_filter_proj=64, num_filter_upsample=64, use_global_stats=use_global_stats)
    hyper1 = mx.sym.concat(inc3e, up43)
    hyper1_0 = conv_bn_relu(hyper1, 'hyper1_0', num_filter=256,
            kernel=(1, 1), pad=(0, 0), use_global_stats=use_global_stats)
    hyper1_1 = conv_bn_relu(hyper1, 'hyper1_1', num_filter=256,
            kernel=(1, 1), pad=(0, 0), use_global_stats=use_global_stats)
    hyper_group.append(mx.sym.concat(hyper1_0, hyper1_1))

    # hyper2
    hyper2 = conv_bn_relu(inc4e, 'hyper2_dilate', num_filter=512,
            kernel=(3, 3), pad=(6, 6), dilate=(6, 6), use_global_stats=use_global_stats)
    hyper2_0 = conv_bn_relu(hyper2, 'hyper2_0', num_filter=256,
            kernel=(1, 1), pad=(0, 0), use_global_stats=use_global_stats)
    hyper2_1 = conv_bn_relu(hyper2, 'hyper2_1', num_filter=256,
            kernel=(1, 1), pad=(0, 0), use_global_stats=use_global_stats)
    hyper_group.append(mx.sym.concat(hyper2_0, hyper2_1))

    # hyper3, 4
    nf_hyper = [256, 192]
    convi = hyper2
    for i, nfh in enumerate(nf_hyper, 3):
        pooli = mx.sym.Pooling(convi, kernel=(2, 2), stride=(2, 2), pool_type='max')
        hyperres = conv_bn_relu(pooli, 'hyperres/{}'.format(i), num_filter=nfh,
                kernel=(1, 1), pad=(0, 0), use_global_stats=use_global_stats)
        hyper1x1 = conv_bn_relu(convi, 'hyper1x1/{}'.format(i), num_filter=nfh/4,
                kernel=(1, 1), pad=(0, 0), use_global_stats=use_global_stats)
        hyper3x3 = conv_bn_relu(hyper1x1, 'hyper3x3/{}'.format(i), num_filter=nfh,
                kernel=(3, 3), pad=(1, 1), stride=(2, 2), use_global_stats=use_global_stats)
        hyperi = hyperres + hyper3x3
        hyperi_0 = conv_bn_relu(hyperi, 'hyper{}_0'.format(i), num_filter=nfh,
                kernel=(1, 1), pad=(0, 0), use_global_stats=use_global_stats)
        hyperi_1 = conv_bn_relu(hyperi, 'hyper{}_1'.format(i), num_filter=nfh,
                kernel=(1, 1), pad=(0, 0), use_global_stats=use_global_stats)
        convi = hyperi
        hyper_group.append(mx.sym.concat(hyperi_0, hyperi_1))

    return hyper_group


def get_symbol(num_classes=1000, **kwargs):
    ''' network for training
    '''
    use_global_stats = False
    if 'use_global_stats' in kwargs:
        use_global_stats = kwargs['use_global_stats']

    data = mx.symbol.Variable(name="data")
    label = mx.symbol.Variable(name="label")

    hyper_group = pvanet_preact(data, use_global_stats)

    pooled = [mx.sym.Pooling(h, kernel=(2, 2), pool_type='max', global_pool=True) for h in hyper_group]

    pooled_all = mx.sym.flatten(mx.sym.concat(*pooled), name='flatten')
    # softmax = mx.sym.SoftmaxOutput(data=pooled_all, label=label, name='softmax')
    fc1 = mx.sym.FullyConnected(pooled_all, num_hidden=4096, name='fc1')
    softmax = mx.sym.SoftmaxOutput(data=fc1, label=label, name='softmax')
    return softmax
