import mxnet as mx
'''
Basic blocks
'''
'''
def data_norm(data, name, eps=1e-05):
    #
    kernel = (3, 3)
    pad = (1, 1)
    ones_ = mx.sym.ones(shape=(1, 1, kernel[0], kernel[1])) / 9.0

    mean_ = mx.sym.mean(data, axis=1, keepdims=True)
    mean_ = mx.sym.Convolution(
        data=mean_,
        num_filter=1,
        weight=ones_,
        kernel=kernel,
        pad=pad,
        no_bias=True)
    var_ = mx.sym.mean(mx.sym.square(data), axis=1, keepdims=True)
    var_ = mx.sym.Convolution(
        data=var_,
        num_filter=1,
        weight=ones_,
        kernel=kernel,
        pad=pad,
        no_bias=True)
    var_ = mx.sym.maximum(var_ - mx.sym.square(mean_), 0.0)
    # var_ = mx.sym.broadcast_maximum(var_ - mx.sym.square(mean_), mx.sym.zeros(shape=(1,)))
    norm_ = mx.sym.sqrt(var_ + eps)
    data_ = mx.sym.broadcast_sub(data, mean_)
    data_ = mx.sym.broadcast_div(data_, norm_)
    data_ = mx.sym.broadcast_add(data_, bias_)
    return data_
'''


def bn_relu_conv(data,
                 prefix_name='',
                 num_filter=0,
                 kernel=(3, 3),
                 pad=(0, 0),
                 stride=(1, 1),
                 use_crelu=False,
                 use_global_stats=False,
                 fix_gamma=False,
                 no_bias=True,
                 get_syms=False):
    #
    assert prefix_name != ''
    conv_name = prefix_name + 'conv'
    bn_name = prefix_name + 'bn'
    syms = {}
    bn_ = mx.sym.BatchNorm(
        data,
        use_global_stats=use_global_stats,
        fix_gamma=fix_gamma,
        name=bn_name)
    syms['bn'] = bn_
    relu_ = mx.sym.Activation(bn_, act_type='relu')
    conv_ = mx.sym.Convolution(
        relu_,
        name=conv_name,
        num_filter=num_filter,
        kernel=kernel,
        pad=pad,
        stride=stride,
        no_bias=no_bias)
    syms['conv'] = conv_
    if use_crelu:
        concat_name = prefix_name + 'concat'
        conv_ = mx.sym.concat(conv_, -conv_, name=concat_name)
        syms['concat'] = conv_
    if get_syms:
        return conv_, syms
    else:
        return conv_


def conv_sep_unit(data,
                  prefix_name='',
                  nf3=0,
                  nf1=0,
                  stride=(1, 1),
                  num_group=1,
                  use_global_stats=False,
                  get_syms=False):
    #
    assert prefix_name != ''
    conv3_name = prefix_name + 'conv3'
    bn3_name = prefix_name + 'bn3'
    syms = {}

    bn3 = mx.sym.BatchNorm(
        data,
        use_global_stats=use_global_stats,
        fix_gamma=False,
        name=bn3_name)
    syms['bn3'] = bn3
    relu3 = mx.sym.Activation(bn3, act_type='relu')
    conv3_weight = mx.sym.var(name=conv3_name+'_weight', lr_mult=1.0, wd_mult=1e-03)
    conv3 = mx.sym.Convolution(
        data=relu3,
        name=conv3_name,
        weight=conv3_weight, 
        num_filter=nf3,
        num_group=num_group,
        kernel=(3, 3),
        pad=(1, 1),
        stride=stride,
        no_bias=True)
    syms['conv3'] = conv3
    conv1, syms1 = bn_relu_conv(
        conv3,
        prefix_name=prefix_name,
        num_filter=nf1,
        kernel=(1, 1),
        pad=(0, 0),
        use_global_stats=use_global_stats,
        get_syms=True)
    syms.update(syms1)
    if get_syms:
        return conv1, syms
    else:
        return conv1


def pool(data, name=None, kernel=(2, 2), stride=(2, 2), pool_type='max'):
    pool_ = mx.sym.Pooling(
        data=data,
        name=name,
        kernel=kernel,
        stride=stride,
        pool_type=pool_type)
    return pool_


'''
Cloning blocks
'''


def clone_conv(data, name, src_layer):
    attrs = src_layer.list_attr()
    no_bias = False
    if 'no_bias' in attrs:
        no_bias = attrs['no_bias'] == 'True'
    inputs = src_layer.get_children()

    if no_bias == False:
        conv = mx.symbol.Convolution(
            name=name, data=data, weight=inputs[1], bias=inputs[2], **attrs)
    else:
        conv = mx.symbol.Convolution(
            name=name, data=data, weight=inputs[1], **attrs)
    return conv


def clone_bn(data, name, src_layer):
    attrs = src_layer.list_attr()
    inputs = src_layer.get_children()

    bn = mx.symbol.BatchNorm(
        name=name, data=data, beta=inputs[2], gamma=inputs[1], **attrs)
    return bn


def clone_bn_relu_conv(data, prefix_name='', src_syms=None):
    assert prefix_name != ''
    conv_name = prefix_name + 'conv'
    bn_name = prefix_name + 'bn'
    bn = clone_bn(data, name=bn_name, src_layer=src_syms['bn'])
    relu_ = mx.sym.Activation(bn, act_type='relu')
    conv_ = clone_conv(relu_, name=conv_name, src_layer=src_syms['conv'])
    if 'concat' in src_syms:
        concat_name = prefix_name + 'concat'
        conv_ = mx.sym.concat(conv_, -conv_, name=concat_name)
    return conv_


def clone_conv_sep_unit(data, prefix_name='', src_syms=None):
    assert prefix_name != ''
    conv3_name = prefix_name + 'conv3'
    bn3_name = prefix_name + 'bn3'

    bn3 = clone_bn(data, bn3_name, src_syms['bn3'])
    relu3 = mx.sym.Activation(bn3, act_type='relu')
    conv3 = clone_conv(relu3, conv3_name, src_syms['conv3'])
    conv = clone_bn_relu_conv(conv3, prefix_name, src_syms)
    return conv


'''
Misc
'''


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
