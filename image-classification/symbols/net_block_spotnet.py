import mxnet as mx
'''
Basic blocks
'''


def pool(data, name=None, kernel=(2, 2), stride=(2, 2), pool_type='max'):
    pool_ = mx.sym.Pooling(
        data=data,
        name=name,
        kernel=kernel,
        stride=stride,
        pool_type=pool_type)
    return pool_


def relu_conv_bn(data, prefix_name='',
                 num_filter=0, kernel=(3, 3), pad=(0, 0), stride=(1, 1), dilate=(1, 1), use_crelu=False,
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
        num_filter=num_filter, kernel=kernel, pad=pad, stride=stride, dilate=dilate,
        no_bias=no_bias)

    syms['conv'] = conv_
    if use_crelu:
        concat_name = prefix_name + 'concat'
        conv_ = mx.sym.concat(conv_, -conv_, name=concat_name)
        syms['concat'] = conv_

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
    if 'concat' in src_syms:
        concat_name = prefix_name + 'concat'
        conv_ = mx.sym.concat(conv_, -conv_, name=concat_name)
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
