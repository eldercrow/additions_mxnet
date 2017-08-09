import mxnet as mx


def convolution(data, name, num_filter, kernel, pad, stride=(1,1), no_bias=False, lr_mult=1.0):
    ''' convolution with lr_mult and wd_mult '''
    w = mx.sym.var(name+'_weight', lr_mult=lr_mult, wd_mult=lr_mult)
    b = None
    if no_bias == False:
        b = mx.sym.var(name+'_bias', lr_mult=lr_mult*2.0, wd_mult=0.0)
    conv = mx.sym.Convolution(data, weight=w, bias=b, name=name, num_filter=num_filter,
            kernel=kernel, pad=pad, stride=stride, no_bias=no_bias)
    return conv


def batchnorm(data, name, use_global_stats, fix_gamma=False, lr_mult=1.0):
    ''' batch norm with lr_mult and wd_mult '''
    g = mx.sym.var(name+'_gamma', lr_mult=lr_mult, wd_mult=0.0)
    b = mx.sym.var(name+'_beta', lr_mult=lr_mult, wd_mult=0.0)
    bn = mx.sym.BatchNorm(data, gamma=g, beta=b, name=name,
            use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    return bn


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


def relu_conv_bn(data, prefix_name, num_filter,
                 kernel=(3,3), pad=(0,0), stride=(1,1), no_bias=True,
                 use_crelu=False,
                 use_global_stats=False, fix_gamma=False,
                 get_syms=False):
    #
    assert prefix_name != ''
    conv_name = prefix_name + 'conv'
    bn_name = prefix_name + 'bn'
    syms = {}
    relu_ = mx.sym.Activation(data, act_type='relu')
    syms['relu'] = relu_

    conv_ = convolution(relu_, conv_name, num_filter, 
            kernel=kernel, pad=pad, stride=stride, no_bias=no_bias)
    syms['conv'] = conv_

    if use_crelu:
        conv_ = mx.sym.concat(conv_, -conv_)
        syms['concat'] = conv_

    bn_ = batchnorm(conv_, bn_name, use_global_stats, fix_gamma)
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
                    num_filter_init=0,
                    use_crelu=False,
                    use_global_stats=False,
                    get_syms=False):
    """
    inception unit, only full padding is supported
    """
    # save symbols anyway
    syms = {}

    prefix_name = prefix_group_name

    if num_filter_init == 0:
        num_filter_init = num_filter_3x3[0]

    bn_, s = relu_conv_bn(data, prefix_name=prefix_name+'init/',
            num_filter=num_filter_init, kernel=(1,1), pad=(0,0),
            use_global_stats=use_global_stats, get_syms=True)
    syms['init'] = bn_

    incep_layers = [bn_] if len(num_filter_3x3) == 0 else []
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
