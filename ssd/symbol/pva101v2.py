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
        proj = conv_bn_relu(data, name+'/proj',
                num_filter=num_filter_proj, kernel=(1, 1), pad=(0, 0),
                use_global_stats=use_global_stats)
    else:
        proj = data
    if num_filter_upsample > 0:
        nf = num_filter_upsample * scale * scale
        relu = conv_bn_relu(proj, name+'/conv',
                num_filter=nf, kernel=(3, 3), pad=(1, 1),
                use_global_stats=use_global_stats)
        return subpixel_upsample(relu, num_filter_upsample, scale, scale)
    else:
        relu = mx.sym.UpSampling(proj, scale=scale, sample_type='bilinear',
                num_filter=num_filter_proj, name=name+'/conv')
        return relu


def downsample_feature(data, name, scale, num_filter_proj, use_global_stats=False):
    ''' Downsample w/ pooling, followed by 1 by 1 projection. '''
    pool = mx.sym.Pooling(data, kernel=(scale, scale), stride=(scale, scale), pool_type='max')
    relu = conv_bn_relu(pool, name+'/conv',
            num_filter=num_filter_proj, kernel=(1, 1), pad=(0, 0),
            use_global_stats=use_global_stats)
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

    groups = [conv3, inc3e, inc4e]
    nf_group = [256, 256, 256]
    convi = inc4e 
    for i, nfg in enumerate(nf_group, 3):
        kernel = (2, 2) if i < 5 else (3, 3)
        conv1x1 = conv_bn_relu(convi, 'conv1x1/{}'.format(i), num_filter=nfg/2,
                kernel=(1, 1), pad=(0, 0), use_global_stats=use_global_stats)
        pad = (1, 1) if i < 5 else (0, 0)
        convi = conv_bn_relu(conv1x1, 'conv3x3/{}'.format(i), num_filter=nfg,
                kernel=(3, 3), pad=pad, stride=(2, 2), use_global_stats=use_global_stats)
        groups.append(convi)

    dn_groups = []
    dn_groups.append([])
    nf_dn = [96, 96, 96, 128, 128]
    scale_dn = [2, 2, 2, 2, 3]
    for i, (g, nfd, ss) in enumerate(zip(groups[:-1], nf_dn, scale_dn)):
        d = downsample_feature(g, name='dn{}{}'.format(i, i+1), scale=ss,
                num_filter_proj=nfd, use_global_stats=use_global_stats)
        dn_groups.append([d])

    up_groups = []
    nf_up = [128, 128, 96, 96, 96]
    scale_up = [2, 2, 2, 2, 3]
    for i, (g, nfu, ss) in enumerate(zip(groups[1:], nf_up, scale_up)):
        u = upsample_feature(g, name='up{}{}'.format(i+1, i), scale=ss,
                num_filter_proj=nfu, use_global_stats=use_global_stats)
        up_groups.append([u])
    up_groups.append([])
    u20 = upsample_feature(groups[2], name='up20', scale=4,
            num_filter_proj=128, use_global_stats=use_global_stats)
    up_groups[0].append(u20)

    hyper_group = []
    for i, (g, d, u) in enumerate(zip(groups, dn_groups, up_groups)):
        hyper = mx.sym.concat(*([g] + u + d), name='hyperconcat{}'.format(i))
        hyper0 = conv_bn_relu(hyper, 'hyper{}_0'.format(i), num_filter=256,
            kernel=(1, 1), pad=(0, 0), use_global_stats=use_global_stats)
        hyper1 = conv_bn_relu(hyper, 'hyper{}_1'.format(i), num_filter=256,
            kernel=(1, 1), pad=(0, 0), use_global_stats=use_global_stats)
        hyper_group.append(mx.sym.concat(hyper0, hyper1))

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
