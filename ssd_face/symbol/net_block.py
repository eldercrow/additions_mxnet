import mxnet as mx

def find_layer(syms, layer_name):
    for s in syms:
        if s.name == layer_name:
            return s

    print 'Could not find the layer %s' % layer_name
    return None

def conv_bn_relu(data, group_name, num_filter, kernel, pad, stride, use_global):
    """ used in inception """
    bn_name = group_name + '_bn'
    relu_name = group_name + '_relu'
    conv_name = group_name + '_conv'

    conv = mx.symbol.Convolution(name=conv_name, data=data, 
            num_filter=num_filter, pad=pad, kernel=kernel, stride=stride, no_bias=True)
    bn = mx.symbol.BatchNorm(name=bn_name, data=conv, use_global_stats=use_global, fix_gamma=False)
    relu = mx.symbol.Activation(name=relu_name, data=bn, act_type='relu')

    syms = {conv_name: conv, bn_name: bn}

    return relu, syms

def bn_relu_conv(data, group_name, num_filter, kernel, pad, stride, use_global, no_bias=False):
    """ used in mCReLU """
    bn_name = group_name + '_bn'
    relu_name = group_name + '_relu'
    conv_name = group_name + '_conv'

    bn = mx.symbol.BatchNorm(name=bn_name, data=data, use_global_stats=use_global, fix_gamma=False)
    relu = mx.symbol.Activation(name=relu_name, data=bn, act_type='relu')
    conv = mx.symbol.Convolution(name=conv_name, data=relu, 
            num_filter=num_filter, pad=pad, kernel=kernel, stride=stride, no_bias=no_bias)

    syms = {conv_name: conv, bn_name: bn}

    return conv, syms

def bn_crelu_conv(data, group_name, num_filter, kernel, pad, stride, use_global):
    """ used in mCReLU """
    concat_name = group_name + '_concat'
    bn_name = group_name + '_bn'
    relu_name = group_name + '_relu'
    conv_name = group_name + '_conv'

    concat = mx.symbol.Concat(name=concat_name, *[data, -data])
    bn = mx.symbol.BatchNorm(name=bn_name, data=concat, use_global_stats=use_global, fix_gamma=False)
    relu = mx.symbol.Activation(name=relu_name, data=bn, act_type='relu')
    conv = mx.symbol.Convolution(name=conv_name, data=relu, 
            num_filter=num_filter, pad=pad, kernel=kernel, stride=stride, no_bias=False)

    return conv

def relu_conv(data, group_name, num_filter, kernel, pad, stride, no_bias=False):
    """ used in mCReLU """
    relu_name = group_name + '_relu'
    conv_name = group_name + '_conv'

    relu = mx.symbol.Activation(name=relu_name, data=data, act_type='relu')
    conv = mx.symbol.Convolution(name=conv_name, data=relu, 
            num_filter=num_filter, pad=pad, kernel=kernel, stride=stride, no_bias=no_bias)

    syms = {conv_name: conv}

    return conv, syms

def clone_conv(data, conv_name, src_layer):
    attrs = src_layer.list_attr()
    no_bias = False
    if 'no_bias' in attrs:
        no_bias = attrs['no_bias']
    inputs = src_layer.get_children()

    if no_bias == False:
        conv = mx.symbol.Convolution(name=conv_name, data=data,
                weight=inputs[1], 
                bias=inputs[2],
                **attrs)
    else:
        conv = mx.symbol.Convolution(name=conv_name, data=data,
                weight=inputs[1], 
                **attrs)
    return conv

def clone_bn(data, bn_name, src_layer):
    attrs = src_layer.list_attr()
    inputs = src_layer.get_children()

    bn = mx.symbol.BatchNorm(name=bn_name, data=data, 
            beta=inputs[2], 
            gamma=inputs[1], 
            **attrs)
    return bn

def clone_conv_bn_relu(data, group_name, src_syms, src_name):
    bn_name = group_name + '_bn'
    relu_name = group_name + '_relu'
    conv_name = group_name + '_conv'

    # clone conv layer
    src_bn_name = src_name + '_bn'
    src_conv_name = src_name + '_conv'

    conv = clone_conv(data, conv_name, src_syms[src_conv_name])
    bn = clone_bn(conv, bn_name, src_syms[src_bn_name])
    relu = mx.symbol.Activation(name=relu_name, data=bn, act_type='relu')

    syms = {conv_name: conv, bn_name: bn}
    return relu, syms

def clone_bn_relu_conv(data, group_name, src_syms, src_name):
    bn_name = group_name + '_bn'
    relu_name = group_name + '_relu'
    conv_name = group_name + '_conv'

    # clone conv layer
    src_bn_name = src_name + '_bn'
    src_conv_name = src_name + '_conv'

    bn = clone_bn(data, bn_name, src_syms[src_bn_name])
    relu = mx.symbol.Activation(name=relu_name, data=bn, act_type='relu')
    conv = clone_conv(relu, conv_name, src_syms[src_conv_name])

    syms = {conv_name: conv, bn_name: bn}
    return conv, syms

def clone_relu_conv(data, group_name, src_syms, src_name):
    relu_name = group_name + '_relu'
    conv_name = group_name + '_conv'

    # clone conv layer
    src_conv_name = src_name + '_conv'

    relu = mx.symbol.Activation(name=relu_name, data=data, act_type='relu')
    conv = clone_conv(relu, conv_name, src_syms[src_conv_name])

    syms = {conv_name: conv}
    return conv, syms

