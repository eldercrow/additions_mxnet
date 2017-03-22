import mxnet as mx

'''
Basic blocks
'''
def bn_relu(data, name, use_global_stats=False, fix_gamma=True):
    #
    bn_ = mx.sym.BatchNorm(data, use_global_stats=use_global_stats, fix_gamma=fix_gamma, name=name)
    relu_ = mx.sym.Activation(bn_, act_type='relu')
    return relu_

def bn_relu_conv(data, prefix_name='', postfix_name='', 
        num_filter=0, kernel=(3,3), dilate=(1,1), stride=(1,1), pad=(0,0), 
        use_global_stats=False, fix_gamma=True, no_bias=True, get_syms=False):
    #
    assert prefix_name != '' or postfix_name != ''
    conv_name = prefix_name + 'conv' + postfix_name
    bn_name = prefix_name + 'bn' + postfix_name

    bn_ = mx.sym.BatchNorm(data, use_global_stats=use_global_stats, fix_gamma=fix_gamma, name=bn_name)
    relu_ = mx.sym.Activation(bn_, act_type='relu')
    conv_ = mx.sym.Convolution(relu_, name=conv_name, num_filter=num_filter, 
            kernel=kernel, dilate=dilate, stride=stride, pad=pad, no_bias=no_bias)
    syms = {'conv': conv_, 'bn': bn_}
    if get_syms:
        return conv_, syms
    else:
        return conv_

def bn_crelu_conv(data, prefix_name='', postfix_name='', 
        num_filter=0, kernel=(3,3), dilate=(1,1), stride=(1,1), pad=(0,0), 
        use_global_stats=False, fix_gamma=True, no_bias=True, get_syms=False): 
    #
    assert prefix_name != '' or postfix_name != ''
    conv_name = prefix_name + 'conv' + postfix_name
    bn_name = prefix_name + 'bn' + postfix_name

    bn_ = mx.sym.BatchNorm(data, use_global_stats=use_global_stats, fix_gamma=fix_gamma, name=bn_name)
    concat_ = mx.sym.concat(bn_, -bn_)
    relu_ = mx.sym.Activation(concat_, act_type='relu')
    conv_ = mx.sym.Convolution(relu_, num_filter=num_filter, 
            kernel=kernel, dilate=dilate, stride=stride, pad=pad, 
            no_bias=no_bias, name=conv_name)
    syms = {'conv': conv_, 'bn': bn_}
    if get_syms:
        return conv_, syms
    else:
        return conv_

def pool(data, name=None, kernel=(2,2), stride=(2,2), pool_type='max'):
    pool_ = mx.sym.Pooling(data=data, name=name, kernel=kernel, stride=stride, pool_type=pool_type)
    return pool_

'''
Cloning blocks
'''
def clone_conv(data, name, src_layer):
    attrs = src_layer.list_attr()
    no_bias = False
    if 'no_bias' in attrs:
        no_bias = attrs['no_bias']
    inputs = src_layer.get_children()

    if no_bias == False:
        conv = mx.symbol.Convolution(name=name, data=data,
                weight=inputs[1], 
                bias=inputs[2],
                **attrs)
    else:
        conv = mx.symbol.Convolution(name=name, data=data,
                weight=inputs[1], 
                **attrs)
    return conv

def clone_bn(data, name, src_layer):
    attrs = src_layer.list_attr()
    inputs = src_layer.get_children()

    bn = mx.symbol.BatchNorm(name=name, data=data, 
            beta=inputs[2], 
            gamma=inputs[1], 
            **attrs)
    return bn

def clone_bn_relu_conv(data, prefix_name='', postfix_name='', src_syms=None):
    assert prefix_name != '' or postfix_name != ''
    conv_name = prefix_name + 'conv' + postfix_name
    bn_name = prefix_name + 'bn' + postfix_name

    bn = clone_bn(data, name=bn_name, src_layer=src_syms['bn'])
    relu_ = mx.sym.Activation(bn, act_type='relu')
    conv_ = clone_conv(relu_, name=conv_name, src_layer=src_syms['conv'])
    return conv_

'''
Misc
'''
def subpixel_upsample(data, ch, c, r):
    if r == 1 and c == 1:
        return data
    X = mx.sym.reshape(data=data, shape=(-3, 0, 0)) # (bsize*ch*r*c, a, b)
    X = mx.sym.reshape(data=X, shape=(-4, -1, r*c, 0, 0)) # (bsize*ch, r*c, a, b)
    X = mx.sym.transpose(data=X, axes=(0, 3, 2, 1)) # (bsize*ch, b, a, r*c)
    X = mx.sym.reshape(data=X, shape=(0, 0, -1, c)) # (bsize*ch, b, a*r, c)
    X = mx.sym.transpose(data=X, axes=(0, 2, 1, 3)) # (bsize*ch, a*r, b, c)
    X = mx.sym.reshape(data=X, shape=(-4, -1, ch, 0, -3)) # (bsize, ch, a*r, b*c)
    return X
