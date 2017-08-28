import mxnet as mx


def conv_bn_relu(data, group_name,
        num_filter, kernel, pad, stride=(1, 1), no_bias=True,
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
    nf = num_filter_upsample * scale * scale
    bn = conv_bn_relu(proj, name+'/conv',
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
        pool1 = mx.sym.Pooling(data, name=prefix_group+'/pool1',
                kernel=(2,2), pad=(0,0), stride=(2,2), pool_type='max')
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


def residual_inc(lhs, rhs, prefix_lhs, prefix_rhs,
        num_filter, stride, no_bias, use_global_stats, get_elt=False):
    ''' residual connection between inception layers '''
    lhs = mx.sym.Convolution(lhs, name=prefix_lhs+'/proj',
            num_filter=num_filter, kernel=(1,1), pad=(0,0), stride=stride, no_bias=no_bias)
    rhs = mx.sym.Convolution(rhs, name=prefix_rhs+'/out',
            num_filter=num_filter, kernel=(1,1), pad=(0,0), stride=(1,1), no_bias=no_bias)
    elt = lhs+rhs
    bn = mx.sym.BatchNorm(elt, name=prefix_rhs+'/elt/bn',
            use_global_stats=use_global_stats, fix_gamma=False)
    relu = mx.sym.Activation(bn, act_type='relu')
    if get_elt:
        return relu, elt
    else:
        return relu


def pvanet_preact(data, use_global_stats=True, no_bias=False):
    ''' pvanet 10.0 '''
    conv1 = conv_bn_relu(data, group_name='conv1',
            num_filter=16, kernel=(4,4), pad=(1,1), stride=(2,2), no_bias=no_bias,
            use_global_stats=use_global_stats, use_crelu=True)
    # conv2
    conv2 = mcrelu(conv1, prefix_group='conv2',
            filters=(16, 24, 48), no_bias=no_bias, use_global_stats=use_global_stats)
    # 64
    conv3 = mcrelu(conv2, prefix_group='conv3',
            filters=(24, 48, 96), no_bias=no_bias, use_global_stats=use_global_stats)
    inc31 = inception(conv3, prefix_group='inc31', # inc3a
            filters_1=64, filters_3=(16,32), filters_5=(16,32,32), no_bias=no_bias,
            use_global_stats=use_global_stats)
    inc32 = inception(inc31, prefix_group='inc32', # inc3b
            filters_1=64, filters_3=(16,32), filters_5=(16,32,32), no_bias=no_bias,
            use_global_stats=use_global_stats)
    inc3 = residual_inc(conv2, inc32, prefix_lhs='conv2', prefix_rhs='inc32',
            num_filter=192, stride=(2, 2), no_bias=no_bias, use_global_stats=use_global_stats)
    # 32
    inc41 = inception(inc3, prefix_group='inc41', # inc3c
            filters_1=96, filters_3=(16,64), filters_5=(16,32,32), no_bias=no_bias,
            use_global_stats=use_global_stats, do_pool=True)
    inc42 = inception(inc41, prefix_group='inc42', # inc3d
            filters_1=96, filters_3=(16,64), filters_5=(16,32,32), no_bias=no_bias,
            use_global_stats=use_global_stats)
    inc43 = inception(inc42, prefix_group='inc43', # inc3e
            filters_1=96, filters_3=(16,64), filters_5=(16,32,32), no_bias=no_bias,
            use_global_stats=use_global_stats)
    inc4 = residual_inc(inc3, inc43, prefix_lhs='inc3', prefix_rhs='inc43',
            num_filter=256, stride=(2, 2), no_bias=no_bias, use_global_stats=use_global_stats)
    # 16
    inc51 = inception(inc4, prefix_group='inc51', # inc4a
            filters_1=96, filters_3=(16,64), filters_5=(16,32,32), no_bias=no_bias,
            use_global_stats=use_global_stats, do_pool=True)
    inc52 = inception(inc51, prefix_group='inc52', # inc4b
            filters_1=128, filters_3=(32,96), filters_5=(16,32,32), no_bias=no_bias,
            use_global_stats=use_global_stats)
    inc53 = inception(inc52, prefix_group='inc53', # inc4c
            filters_1=128, filters_3=(32,96), filters_5=(16,32,32), no_bias=no_bias,
            use_global_stats=use_global_stats)
    inc5 = residual_inc(inc4, inc53, prefix_lhs='inc4', prefix_rhs='inc53',
            num_filter=256, stride=(2,2), no_bias=no_bias, use_global_stats=use_global_stats)
    # 8
    inc61 = inception(inc5, prefix_group='inc61', # inc4d
            filters_1=128, filters_3=(32,96), filters_5=(16,32,32), no_bias=no_bias,
            use_global_stats=use_global_stats, do_pool=True)
    inc62 = inception(inc61, prefix_group='inc62', # inc4e
            filters_1=128, filters_3=(32,96), filters_5=(16,32,32), no_bias=no_bias,
            use_global_stats=use_global_stats)
    inc6 = residual_inc(inc5, inc62, prefix_lhs='inc5', prefix_rhs='inc62',
            num_filter=256, stride=(2,2), no_bias=no_bias, use_global_stats=use_global_stats)
    # 4
    inc71 = inception(inc6, prefix_group='inc71',
            filters_1=128, filters_3=(32,96), filters_5=(16,32,32), no_bias=no_bias,
            use_global_stats=use_global_stats, do_pool=True)
    inc7 = residual_inc(inc6, inc71, prefix_lhs='inc6', prefix_rhs='inc71',
            num_filter=256, stride=(2,2), no_bias=no_bias, use_global_stats=use_global_stats)
    # 1
    conv81 = mx.sym.Convolution(inc7, name='conv81',
            num_filter=256, kernel=(4,4), pad=(0,0), stride=(1,1), no_bias=no_bias)
    pool82 = mx.sym.Pooling(inc7, kernel=(4, 4), pool_type='max')
    conv82 = mx.sym.Convolution(pool82, name='conv82',
            num_filter=256, kernel=(1,1), pad=(0,0), no_bias=no_bias)
    elt8 = conv81 + conv82
    bn8 = mx.sym.BatchNorm(elt8, name='bn8', use_global_stats=use_global_stats, fix_gamma=False)
    relu8 = mx.sym.Activation(bn8, act_type='relu')

    # channels: (192, 256, 256, 256, 256, 256)
    return [inc3, inc4, inc5, inc6, inc7, relu8]


def downsample_groups(groups, use_global_stats):
    dn_groups = [[] for _ in groups]
    for i, g in enumerate(groups[1:-2], 2):
        d = conv_bn_relu(g, 'dn{}'.format(i),
                num_filter=64, kernel=(3, 3), pad=(1, 1), stride=(2, 2),
                use_global_stats=use_global_stats)
        dn_groups[i].append(d)
    dn_groups[-1].append( \
            conv_bn_relu(groups[-2], 'dn{}'.format(len(groups)-1),
                    num_filter=128, kernel=(4, 4), pad=(0, 0),
                    use_global_stats=use_global_stats))
    return dn_groups


def upsample_groups(groups, use_global_stats):
    up_groups = [[] for _ in groups]
    for i, g in enumerate(groups[1:3], 1):
        u = upsample_feature(g, 'up{}{}'.format(i, i-1), scale=2,
                num_filter_proj=64, num_filter_upsample=64,
                use_global_stats=use_global_stats)
        up_groups[i-1].append(u)
    up_groups[0].append( \
            upsample_feature(groups[2], 'up20', scale=4,
                num_filter_proj=128, num_filter_upsample=64,
                use_global_stats=use_global_stats))
    up_groups.append([])
    return up_groups


def get_symbol(num_classes=1000, **kwargs):
    ''' network for training
    '''
    use_global_stats = False
    if 'use_global_stats' in kwargs:
        use_global_stats = kwargs['use_global_stats']

    data = mx.symbol.Variable(name="data")
    label = mx.symbol.Variable(name="label")

    groups = pvanet_preact(data, use_global_stats)
    dn_groups = downsample_groups(groups, use_global_stats)
    up_groups = upsample_groups(groups, use_global_stats)

    nf_hyper=320

    hyper_groups = []
    for i, (g, u, d) in enumerate(zip(groups, up_groups, dn_groups)):
        h = mx.sym.concat(*([g] + d + u))
        hc = conv_bn_relu(h, 'hyper{}'.format(i),
                num_filter=nf_hyper, kernel=(1, 1), pad=(0, 0),
                use_global_stats=use_global_stats)
        hyper_groups.append(hc)

    pooled = []
    for i, h in enumerate(hyper_groups):
        p = mx.sym.Pooling(h, kernel=(2,2), global_pool=True, pool_type='max')
        pooled.append(p)

    pooled_all = mx.sym.flatten(mx.sym.concat(*pooled), name='flatten')
    # softmax = mx.sym.SoftmaxOutput(data=pooled_all, label=label, name='softmax')
    fc1 = mx.sym.FullyConnected(pooled_all, num_hidden=4096, name='fc1')
    softmax = mx.sym.SoftmaxOutput(data=fc1, label=label, name='softmax')
    return softmax
