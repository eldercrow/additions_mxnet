import mxnet as mx
'''
Basic blocks
'''


def data_norm(data, name, nch, bias=None, eps=1e-05, get_syms=False):
    bias_name = name + '_beta'
    if bias:
        bias_ = bias
    else:
        bias_ = mx.sym.var(name=bias_name, shape=(1, nch, 1, 1))

    kernel = (3, 3)
    pad = (1, 1)
    ones_ = mx.sym.ones(shape=(1, 1, kernel[0], kernel[1])) / 9.0

    mean_ = mx.sym.mean(data, axis=1, keepdims=True)
    mean_ = mx.sym.Pooling(
        mean_, kernel=(3, 3), pad=(1, 1), stride=(1, 1), pool_type='avg')
    var_ = mx.sym.mean(mx.sym.square(data), axis=1, keepdims=True)
    var_ = mx.sym.Pooling(
        var_, kernel=(3, 3), pad=(1, 1), stride=(1, 1), pool_type='avg')
    var_ = mx.sym.maximum(var_ - mx.sym.square(mean_), 0.0)
    # var_ = mx.sym.broadcast_maximum(var_ - mx.sym.square(mean_), mx.sym.zeros(shape=(1,)))
    norm_ = mx.sym.sqrt(var_ + eps)
    data_ = mx.sym.broadcast_sub(data, mean_)
    data_ = mx.sym.broadcast_div(data_, norm_)
    data_ = mx.sym.broadcast_add(data_, bias_)

    if get_syms:
        syms = {'bias': bias_}
        return data_, syms
    else:
        return data_


def pool(data, name=None, kernel=(2, 2), stride=(2, 2), pool_type='max'):
    pool_ = mx.sym.Pooling(
        data=data,
        name=name,
        kernel=kernel,
        stride=stride,
        pool_type=pool_type)
    return pool_


def relu_conv_bn(data, prefix_name='',
                 num_filter=0, kernel=(3, 3), pad=(0, 0), stride=(1, 1), dilate=(1, 1),
                 use_global_stats=False, fix_gamma=False, no_bias=True,
                 get_syms=False):
    #
    assert prefix_name != ''
    conv_name = prefix_name + 'conv'
    bn_name = prefix_name + 'bn'
    syms = {}
    relu_ = mx.sym.Activation(data, act_type='relu')
    syms['relu'] = relu_

    conv_ = mx.sym.Convolution(relu_, name=conv_name,
        num_filter=num_filter, kernel=kernel, pad=pad, stride=stride,
        no_bias=no_bias)
    syms['conv'] = conv_

    bn_ = mx.sym.BatchNorm(conv_, name=bn_name, 
            use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    syms['bn'] = bn_

    if get_syms:
        return bn_, syms
    else:
        return bn_


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


def clone_in(data, name, src_layer):
    attrs = src_layer.list_attr()
    inputs = src_layer.get_children()

    bn = mx.symbol.InstanceNorm(
        name=name, data=data, beta=inputs[2], gamma=inputs[1], **attrs)
    return bn


def clone_relu_conv_bn(data, prefix_name='', src_syms=None):
    assert prefix_name != ''
    conv_name = prefix_name + 'conv'
    bn_name = prefix_name + 'bn'
    relu_ = mx.sym.Activation(data, act_type='relu')
    conv_ = clone_conv(relu_, name=conv_name, src_layer=src_syms['conv'])
    bn_ = clone_bn(conv_, name=bn_name, src_layer=src_syms['bn'])
    return bn_


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
