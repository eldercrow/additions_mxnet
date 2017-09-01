import mxnet as mx


def pool(data, name=None, kernel=(2, 2), stride=(2, 2), pool_type='max'):
    return mx.sym.Pooling(data, name=name, kernel=kernel, stride=stride, pool_type=pool_type)


def subpixel_upsample(data, ch, c, r, name=None):
    if r == 1 and c == 1:
        return data
    X = mx.sym.reshape(data=data, shape=(-3, 0, 0))  # (bsize*ch*r*c, a, b)
    X = mx.sym.reshape(
        data=X, shape=(-4, -1, r * c, 0, 0))  # (bsize*ch, r*c, a, b)
    X = mx.sym.transpose(data=X, axes=(0, 3, 2, 1))  # (bsize*ch, b, a, r*c)
    X = mx.sym.reshape(data=X, shape=(0, 0, -1, c))  # (bsize*ch, b, a*r, c)
    X = mx.sym.transpose(data=X, axes=(0, 2, 1, 3))  # (bsize*ch, a*r, b, c)
    X = mx.sym.reshape(data=X, name=name, shape=(-4, -1, ch, 0, -3))  # (bsize, ch, a*r, b*c)
    return X


def bn_relu_conv(data, prefix_name, num_filter,
                 kernel=(1,1), pad=(0,0), stride=(1,1), no_bias=True,
                 use_crelu=False,
                 use_global_stats=False):
    #
    assert prefix_name
    bn_name = prefix_name + 'bn'
    relu_name = prefix_name + 'relu'
    conv_name = prefix_name + 'conv'

    bn_ = mx.sym.BatchNorm(data, name=bn_name, use_global_stats=use_global_stats, fix_gamma=False)
    relu_ = mx.sym.Activation(bn_, act_type='relu')
    conv_ = mx.sym.Convolution(relu_, name=conv_name, num_filter=num_filter,
            kernel=kernel, pad=pad, stride=stride, no_bias=no_bias)
    if use_crelu:
        conv_ = mx.sym.concat(conv_, -conv_)
    return conv_


def conv_group(data,
               prefix_name,
               num_filter_3x3,
               num_filter_1x1=0,
               do_proj=False,
               use_crelu=False,
               use_global_stats=False):
    '''
    '''
    cgroup = []

    if num_filter_1x1 > 0:
        conv_ = bn_relu_conv(data, prefix_name=prefix_name+'init/',
                num_filter=num_filter_1x1, kernel=(1, 1), pad=(0, 0),
                use_global_stats=use_global_stats)
        cgroup.append(conv_)
    else:
        conv_ = data

    for ii, nf3 in enumerate(num_filter_3x3):
        conv_ = bn_relu_conv(
            conv_, prefix_name=prefix_name + '3x3/{}/'.format(ii),
            num_filter=nf3, kernel=(3,3), pad=(1,1), use_crelu=use_crelu,
            use_global_stats=use_global_stats)
        cgroup.append(conv_)

    concat_ = mx.sym.concat(*cgroup, name=prefix_name + 'concat/')
    if do_proj:
        nf_proj = num_filter_1x1 + sum(num_filter_3x3)
        concat_ = bn_relu_conv(concat_, prefix_name=prefix_name+'proj/',
                num_filter=nf_proj, kernel=(1, 1), pad=(0, 0),
                use_global_stats=use_global_stats)
    return concat_


def proj_add(lhs, rhs, num_filter, use_global_stats):
    lhs = bn_relu_conv(lhs, prefix_name=lhs.name+'proj/',
            num_filter=num_filter, kernel=(1, 1), pad=(0, 0),
            use_global_stats=use_global_stats)
    rhs = bn_relu_conv(rhs, prefix_name=rhs.name+'proj/',
            num_filter=num_filter, kernel=(1, 1), pad=(0, 0),
            use_global_stats=use_global_stats)
    return lhs + rhs


def upsample_feature(data,
                     name,
                     scale,
                     num_filter_proj=0,
                     num_filter_upsample=0,
                     use_global_stats=False):
    ''' use subpixel_upsample to upsample a given layer '''
    if num_filter_proj > 0:
        proj = bn_relu_conv(data, prefix_name=name+'proj/',
                num_filter=num_filter_proj, kernel=(1, 1), pad=(0, 0),
                use_global_stats=use_global_stats)
    else:
        proj = data
    if num_filter_upsample > 0:
        nf = num_filter_upsample * scale * scale
        conv = bn_relu_conv(proj, prefix_name=name+'conv/',
                num_filter=nf, kernel=(3, 3), pad=(1, 1),
                use_global_stats=use_global_stats)
        return subpixel_upsample(conv, num_filter_upsample, scale, scale, name=name+'subpixel')
    else:
        conv = mx.sym.UpSampling(proj, name=name+'conv/', scale=scale,
                num_filter=num_filter_proj, sample_type='bilinear')
        return conv


def proj_add(lhs, rhs, name, num_filter, use_global_stats):
    ''' 1x1 convolution followed by elewise add. '''
    lhs = bn_relu_conv(lhs, name+'lhs/', num_filter=num_filter,
            kernel=(1, 1), pad=(0, 0), use_global_stats=use_global_stats)
    rhs = bn_relu_conv(rhs, name+'rhs/', num_filter=num_filter,
            kernel=(1, 1), pad=(0, 0), use_global_stats=use_global_stats)
    return lhs + rhs


def multiple_conv(data,
                  prefix_name,
                  num_filter_3x3,
                  num_filter_1x1=0,
                  use_crelu=False,
                  use_global_stats=False):
    '''
    '''
    if num_filter_1x1 > 0:
        conv_ = bn_relu_conv(data, prefix_name=prefix_name+'init/',
                num_filter=num_filter_1x1, kernel=(1,1), pad=(0,0),
                use_global_stats=use_global_stats)
    else:
        conv_ = data

    for ii, nf3 in enumerate(num_filter_3x3):
        conv_ = bn_relu_conv(
            conv_, prefix_name=prefix_name + '3x3/{}/'.format(ii),
            num_filter=nf3, kernel=(3,3), pad=(1,1), use_crelu=use_crelu,
            use_global_stats=use_global_stats)

    return conv_
