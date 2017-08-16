import mxnet as mx
import numpy as np


def pool(data, name=None, kernel=(2, 2), stride=(2, 2), pool_type='max'):
    return mx.sym.Pooling(data, name=name, kernel=kernel, stride=stride, pool_type=pool_type)


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


def relu_conv_bn(data, prefix_name='',
                 num_filter=0, kernel=(3, 3), pad=(0, 0), stride=(1, 1), use_crelu=False,
                 use_global_stats=False, fix_gamma=False, no_bias=True,
                 get_syms=False):
    #
    assert prefix_name != ''
    conv_name = prefix_name + 'conv'
    bn_name = prefix_name + 'bn'
    syms = {}

    relu_ = mx.sym.LeakyReLU(data, act_type='leaky')
    # relu_ = mx.sym.Activation(data, act_type='softrelu')
    syms['relu'] = relu_

    conv_ = mx.sym.Convolution(relu_, name=conv_name,
            num_filter=num_filter, kernel=kernel, pad=pad, stride=stride, no_bias=no_bias)
    syms['conv'] = conv_

    if use_crelu:
        concat_name = prefix_name + 'concat'
        conv_ = mx.sym.concat(conv_, -conv_, name=concat_name)
        syms['concat'] = conv_

    bn_ = mx.sym.BatchNorm(conv_, name=bn_name, use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    syms['bn'] = bn_

    if get_syms:
        return bn_, syms
    else:
        return bn_


def inception_group(data,
                    prefix_group_name,
                    n_curr_ch,
                    num_filter_3x3,
                    num_filter_1x1,
                    use_crelu=False,
                    use_global_stats=False,
                    get_syms=False):
    """
    inception unit, only full padding is supported
    """
    # save symbols anyway
    syms = {}

    prefix_name = prefix_group_name

    bn_, s = relu_conv_bn(data, prefix_name=prefix_name+'init/',
            num_filter=num_filter_3x3[0], kernel=(1,1), pad=(0,0),
            use_global_stats=use_global_stats, get_syms=True)
    syms['init'] = bn_

    incep_layers = [bn_]
    for ii, nf3 in enumerate(num_filter_3x3):
        bn_, s = relu_conv_bn(
            bn_, prefix_name=prefix_name + '3x3/{}/'.format(ii),
            num_filter=nf3, kernel=(3,3), pad=(1,1), use_crelu=use_crelu,
            use_global_stats=use_global_stats, get_syms=True)
        syms['unit{}'.format(ii)] = s
        incep_layers.append(bn_)

    concat_ = mx.sym.concat(*incep_layers)
    proj_, s = relu_conv_bn(concat_,
            prefix_name=prefix_name + '1x1/',
            num_filter=num_filter_1x1, kernel=(1,1),
            use_global_stats=use_global_stats, get_syms=True)
    syms['concat'] = s

    if num_filter_1x1 != n_curr_ch:
        data, s = relu_conv_bn(data, prefix_name=prefix_name + 'proj/',
            num_filter=num_filter_1x1, kernel=(1,1),
            use_global_stats=use_global_stats, get_syms=True)
        syms['proj_data'] = s

    res_ = proj_ + data

    if get_syms:
        return res_, num_filter_1x1, syms
    else:
        return res_, num_filter_1x1


def mcrelu_group(data, prefix_name,
        num_filter_proj, num_filter_3x3, num_filter_1x1,
        use_global_stats=False, get_syms=False):
    '''
    '''
    syms = {}
    if num_filter_proj > 0:
        proj, s = relu_conv_bn(data, prefix_name=prefix_name + 'proj/',
                num_filter=num_filter_proj, kernel=(1, 1), pad=(0, 0),
                use_global_stats=use_global_stats, fix_gamma=False, get_syms=True)
        syms['proj'] = s
    else:
        proj = data

    cconv, s = relu_conv_bn(proj, prefix_name=prefix_name + '3x3/',
            num_filter=num_filter_3x3, kernel=(3, 3), pad=(1, 1), use_crelu=True,
            use_global_stats=use_global_stats, fix_gamma=False, get_syms=True)
    syms['cconv'] = s

    conv, s = relu_conv_bn(cconv, prefix_name=prefix_name + '1x1/',
            num_filter=num_filter_1x1, kernel=(1, 1), pad=(0, 0),
            use_global_stats=use_global_stats, fix_gamma=False, get_syms=True)
    syms['conv'] = s

    if get_syms:
        return conv, syms
    else:
        return conv


def upsample_feature(data,
                     name,
                     scale,
                     num_filter_proj=0,
                     num_filter_upsample=0,
                     use_global_stats=False):
    ''' use subpixel_upsample to upsample a given layer '''
    if num_filter_proj > 0:
        proj = relu_conv_bn(
            data,
            prefix_name=name + 'proj/',
            num_filter=num_filter_proj,
            kernel=(1, 1),
            pad=(0, 0),
            use_global_stats=use_global_stats)
    else:
        proj = data

    nf = num_filter_upsample * scale * scale
    bn = relu_conv_bn(
        proj,
        prefix_name=name + 'conv/',
        num_filter=nf,
        kernel=(3, 3),
        pad=(1, 1),
        use_global_stats=use_global_stats)
    return subpixel_upsample(bn, num_filter_upsample, scale, scale)


def get_symbol(num_classes=1000, **kwargs):
    '''
    '''
    data = mx.sym.var(name='data')
    label = mx.sym.var(name='label')

    use_global_stats=kwargs['use_global_stats']

    conv1 = mx.sym.Convolution(data, name='1/conv',
            num_filter=12, kernel=(3,3), pad=(1,1), no_bias=True)
    concat1 = mx.sym.concat(conv1, -conv1, name='concat1')
    bn1 = mx.sym.BatchNorm(concat1, name='1/bn', use_global_stats=use_global_stats, fix_gamma=False)
    pool1 = pool(bn1)

    n_curr_ch = 24
    bn2, n_curr_ch = inception_group(pool1, '2/', n_curr_ch,
            num_filter_3x3=(16, 16), num_filter_1x1=64, use_crelu=True,
            use_global_stats=use_global_stats)
    pool2 = pool(bn2)

    bn3, n_curr_ch = inception_group(pool2, '3/', n_curr_ch,
            num_filter_3x3=(48, 24, 24), num_filter_1x1=96,
            use_global_stats=use_global_stats)

    # initial forward feature

    # 48, 24, 12, 6, 3
    nf_3x3 = [(64, 32, 32), (96, 48, 48), (128, 64, 64), (64, 32, 32), (64, 64)]
    nf_1x1 = [np.sum(np.array(i)) for i in nf_3x3]

    group_i = bn3
    groups = []
    for i, (nf3, nf1) in enumerate(zip(nf_3x3, nf_1x1)):
        group_i = pool(group_i)
        group_i, n_curr_ch = inception_group(group_i, 'g{}/'.format(i), n_curr_ch,
                num_filter_3x3=nf3, num_filter_1x1=nf1,
                use_global_stats=use_global_stats)
        groups.append(group_i)
    # 1
    group_i = mx.sym.Pooling(group_i, kernel=(3,3), global_pool=True, pool_type='max')
    group_i, n_curr_ch = inception_group(group_i, 'g5/', n_curr_ch,
            num_filter_3x3=(128,), num_filter_1x1=128,
            use_global_stats=use_global_stats)
    groups.append(group_i)

    # upsample direction
    upscale = (2, 2, 2, 2, 3)
    nf_up = (128, 64, 64, 64, 64)
    upgroups = []

    for i, (s, u, g) in enumerate(zip(upscale, nf_up, groups[1:])):
        up = upsample_feature(g, name='gu{}{}/'.format(i+1,i), scale=s,
                num_filter_proj=64, num_filter_upsample=u,
                use_global_stats=use_global_stats)
        upgroups.append([up])
    upgroups[0].append( \
            upsample_feature(groups[2], name='gu20/', scale=4,
                    num_filter_proj=64, num_filter_upsample=64,
                    use_global_stats=use_global_stats))

    upgroups.append([])

    # downsample direction
    lhs = [bn3] + groups[:-1]
    downpads = (1, 1, 1, 1, 1, 0)
    downgroups = []

    for i, (d, g) in enumerate(zip(downpads, lhs)):
        down = relu_conv_bn(g, prefix_name='gd{}/'.format(i),
                num_filter=64, kernel=(3,3), pad=(d,d), stride=(2,2),
                use_global_stats=use_global_stats)
        downgroups.append(down)

    hypergroups = []
    nf_hyper = (384, 384, 256, 256, 256, 192)
    for i, (gg, uu, dd, nf) in enumerate(zip(groups, upgroups, downgroups, nf_hyper)):
        hh = uu + [gg] + [dd]
        g = mx.sym.concat(*hh, name='gud{}/'.format(i))
        g = relu_conv_bn(g, prefix_name='hp{}/'.format(i),
                num_filter=nf, kernel=(1,1),
                use_global_stats=use_global_stats)
        g = mx.sym.LeakyReLU(g, name='hyper{}'.format(i), act_type='leaky')
        hypergroups.append(g)

    pooled = []
    for g in hypergroups:
        pooled.append(mx.sym.Pooling(g, global_pool=True, kernel=(2,2), pool_type='max'))

    return mx.sym.concat(*pooled)
