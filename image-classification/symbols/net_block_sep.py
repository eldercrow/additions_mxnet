import mxnet as mx


def pool(data, name=None, kernel=(2, 2), pad=(0, 0), stride=(2, 2), pool_type='max'):
    return mx.sym.Pooling(data, name=name, kernel=kernel, pad=pad, stride=stride, pool_type=pool_type)


def relu_conv_bn(data, prefix_name,
                 num_filter, kernel=(3, 3), pad=(1, 1), stride=(1, 1), num_group=1,
                 wd_mult=1.0, no_bias=True, use_crelu=False,
                 use_global_stats=False):
    #
    assert prefix_name != ''
    conv_name = prefix_name + 'conv'
    bn_name = prefix_name + 'bn'

    relu = mx.sym.Activation(data, act_type='relu')

    conv_w = mx.sym.var(name=conv_name+'_weight', lr_mult=1.0, wd_mult=wd_mult)
    conv_b = None if no_bias else mx.sym.var(name=conv_name+'_bias', lr_mult=2.0, wd_mult=0.0)
    conv = mx.sym.Convolution(relu, name=conv_name, weight=conv_w, bias=conv_b,
            num_filter=num_filter, kernel=kernel, pad=pad, stride=stride, num_group=num_group,
            no_bias=no_bias)
    if use_crelu:
        conv = mx.sym.concat(conv, -conv)

    bn = mx.sym.BatchNorm(conv, name=bn_name,
            use_global_stats=use_global_stats, fix_gamma=False, eps=1e-04)
    return bn


def depthwise_conv(data, name, nf_dw, nf_sep=0,
        kernel=(3, 3), pad=(1, 1), stride=(1, 1),
        use_global_stats=False):
    #
    if nf_sep == 0:
        nf_sep = nf_dw
    bn_dw = relu_conv_bn(data, name+'dw/',
            num_filter=nf_dw, kernel=kernel, pad=pad, stride=stride, num_group=nf_dw, wd_mult=0.01,
            use_global_stats=use_global_stats)
    bn_sep = relu_conv_bn(bn_dw, name+'sep/',
            num_filter=nf_sep, kernel=(1, 1), pad=(0, 0),
            use_global_stats=use_global_stats)
    return bn_sep


def conv_group(data, prefix_name, nf_dw_group, nf_sep_group, do_proj=False, use_global_stats=False):
    #
    cgroup = []

    bn = data
    for i, (nf_dw, nf_sep) in enumerate(zip(nf_dw_group, nf_sep_group)):
        bn = depthwise_conv(bn, prefix_name+'conv{}/'.format(i),
                nf_dw=nf_dw, nf_sep=nf_sep,
                use_global_stats=use_global_stats)
        cgroup.append(bn)

    concat = mx.sym.concat(*cgroup)
    if do_proj:
        nf_proj = sum(nf_sep_group)
        concat = relu_conv_bn(concat, prefix_name+'proj/',
                num_filter=nf_proj, kernel=(1, 1), pad=(0, 0),
                use_global_stats=use_global_stats)
    return concat


def subpixel_downsample(data, ch, c, r, name=None):
    '''
    '''
    if r == 1 and c == 1:
        return data
    # data = (n, ch, h*r, w*c)
    X = mx.sym.transpose(data, axes=(0, 3, 2, 1)) # (n, w*c, h*r, ch)
    X = mx.sym.reshape(X, shape=(0, 0, -1, r*ch)) # (n, w*c, h, r*ch)
    X = mx.sym.transpose(X, axes=(0, 2, 1, 3)) # (n, h, w*c, r*ch)
    X = mx.sym.reshape(X, shape=(0, 0, -1, r*c*ch)) # (n, h, w, r*c*ch)
    X = mx.sym.transpose(X, axes=(0, 3, 1, 2))
    return X


def subpixel_upsample(data, ch, c, r, name=None):
    '''
    Transform input data shape of (n, ch*r*c, h, w) to (n, ch, h*r, c*w).

    ch: number of channels after upsample
    r: row scale factor
    c: column scale factor
    '''
    if r == 1 and c == 1:
        return data
    X = mx.sym.reshape(data=data, shape=(-3, 0, 0))  # (n*ch*r*c, h, w)
    X = mx.sym.reshape(
        data=X, shape=(-4, -1, r * c, 0, 0))  # (n*ch, r*c, h, w)
    X = mx.sym.transpose(data=X, axes=(0, 3, 2, 1))  # (n*ch, w, h, r*c)
    X = mx.sym.reshape(data=X, shape=(0, 0, -1, c))  # (n*ch, w, h*r, c)
    X = mx.sym.transpose(data=X, axes=(0, 2, 1, 3))  # (n*ch, h*r, w, c)
    X = mx.sym.reshape(data=X, name=name, shape=(-4, -1, ch, 0, -3))  # (n, ch, h*r, w*c)
    return X
