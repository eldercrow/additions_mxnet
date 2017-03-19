# import find_mxnet
import mxnet as mx

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

def conv_subpixel(data, name, num_filter, 
        kernel=(3,3), dilate=(2,2), pad=(0,0), pool_type='max', no_bias=True):
    # no stride accepted, at least for now
    if dilate[0] == 1 and dilate[1] == 1: # ordinary conv
        return mx.sym.Convolution(data=data, name=name, num_filter=num_filter, 
                kernel=kernel, pad=pad, no_bias=no_bias)

    multiplier = dilate[0]*dilate[1]
    # pooling
    pool_ = pool(data, kernel=dilate, stride=dilate, pool_type=pool_type)
    # conv
    n_filter_pooled = num_filter * multiplier
    wd_mult = 1.0 / multiplier
    conv_ = mx.sym.Convolution(data=pool_, name=name, num_filter=n_filter_pooled, 
            attr={'__wd_mult__': str(wd_mult)}, 
            kernel=kernel, pad=(pad[0]/dilate[0], pad[1]/dilate[1]), no_bias=no_bias)
    # subpixel recover
    return subpixel_upsample(conv_, num_filter, dilate[0], dilate[1])

def bn_relu_conv(data, num_filter, prefix_name=None, postfix_name=None, 
        kernel=(3,3), dilate=(1,1), stride=(1,1), pad=(0,0), 
        use_global_stats=False, fix_gamma=True, no_bias=True):
    #
    if prefix_name is None and postfix_name is None:
        conv_name = None
        bn_name = None
    else:
        pr = prefix_name if prefix_name is not None else ''
        po = postfix_name if postfix_name is not None else ''
        conv_name = pr + 'conv' + po
        bn_name = pr + 'bn' + po

    bn_ = mx.sym.BatchNorm(data, use_global_stats=use_global_stats, fix_gamma=fix_gamma, name=bn_name)
    relu_ = mx.sym.Activation(bn_, act_type='relu')
    conv_ = conv_subpixel(relu_, name=conv_name, num_filter=num_filter, 
            kernel=kernel, dilate=dilate, pad=pad, pool_type='max', no_bias=no_bias)
    return conv_

def bn_crelu_conv(data, num_filter, prefix_name=None, postfix_name=None, 
        kernel=(3,3), dilate=(1,1), stride=(1,1), pad=(0,0), 
        use_global_stats=False, fix_gamma=True, no_bias=True): 
    #
    if prefix_name is None and postfix_name is None:
        conv_name = None
        bn_name = None
    else:
        pr = prefix_name if prefix_name is not None else ''
        po = postfix_name if postfix_name is not None else ''
        conv_name = pr + 'conv' + po
        bn_name = pr + 'bn' + po
        # sigma_name = pr + 'dn' + po + '_beta'

    bn_ = mx.sym.BatchNorm(data, use_global_stats=use_global_stats, fix_gamma=fix_gamma, name=bn_name)
    concat_ = mx.sym.concat(bn_, -bn_)
    relu_ = mx.sym.Activation(concat_, act_type='relu')
    conv_ = conv_subpixel(relu_, name=conv_name, num_filter=num_filter, 
            kernel=kernel, dilate=dilate, pad=pad, pool_type='max', no_bias=no_bias)
    return conv_

def bn_relu(data, name, use_global_stats=False, fix_gamma=True):
    bn_ = mx.sym.BatchNorm(data, use_global_stats=use_global_stats, fix_gamma=fix_gamma, name=name)
    relu_ = mx.sym.Activation(bn_, act_type='relu')
    return relu_

def pool(data, kernel=(2,2), stride=(2,2), pool_type='max'):
    pool_ = mx.sym.Pooling(data=data, kernel=kernel, stride=stride, pool_type=pool_type)
    return pool_

def inception_group(data, prefix_group_name, n_curr_ch, 
        num_filter_3x3, num_filter_1x1, 
        use_global_stats=False, fix_gamma=True):
    """ 
    inception unit, only full padding is supported
    """
    prefix_name = prefix_group_name + '/'
    incep_layers = []
    conv_ = bn_relu_conv(data, num_filter_3x3, prefix_name, '3x3', 
            kernel=(3,3), dilate=(1,1), pad=(1,1), 
            use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    convn_ = bn_relu_conv(conv_, num_filter_3x3, prefix_name, '3x3/1', 
            kernel=(1,1), 
            use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    # downsample conv
    convd_ = bn_relu_conv(conv_, num_filter_3x3, prefix_name, '3x3/2', 
            kernel=(3,3), dilate=(2,2), pad=(2,2), 
            use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    # upsample conv
    reluu_ = bn_relu(conv_, prefix_name+'bn3x3/3', 
            use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    convu_ = mx.sym.Convolution(reluu_, num_filter=num_filter_3x3*4, name=prefix_name+'conv3x3/3', 
            kernel=(3,3), dilate=(1,1), pad=(1,1),
            attr={'__wd_mult__': '0.25'}, no_bias=True)
    convu_ = subpixel_upsample(convu_, num_filter_3x3, 2, 2)
    convu_ = pool(convu_, kernel=(2,2), stride=(2,2), pool_type='max')

    concat_ = mx.sym.concat(convn_, convd_, convu_)

    concat_ = bn_relu_conv(concat_, num_filter_1x1, prefix_name, '1x1/c', 
            kernel=(1,1), 
            use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    
    if n_curr_ch != num_filter_1x1:
        data = bn_relu_conv(data, num_filter_1x1, prefix_name+'proj/', 
                kernel=(1,1), 
                use_global_stats=use_global_stats, fix_gamma=fix_gamma)

    return concat_ + data, num_filter_1x1

def get_hjnet_conv(data, is_test, fix_gamma=True):
    """ main shared conv layers """
    data_ = data * (1.0 / 255.0)
    conv1_1 = mx.sym.Convolution(data_, name='conv1/1', num_filter=16,  
            kernel=(3,3), pad=(0,0), no_bias=True) # 32, 198
    conv1_2 = bn_crelu_conv(conv1_1, 32, postfix_name='1/2', 
            kernel=(3,3), pad=(0,0), use_global_stats=is_test, fix_gamma=fix_gamma) # 48, 196 
    conv1_3 = bn_crelu_conv(conv1_2, 32, postfix_name='1/3', 
            kernel=(3,3), dilate=(2,2), pad=(0,0), use_global_stats=is_test, fix_gamma=fix_gamma) # 48, 192 
    crop1_2 = mx.sym.Crop(conv1_2, conv1_3, center_crop=True)
    conv1_3 = crop1_2 + conv1_3

    nf_3x3 = [32, 48, 64, 72, 96]
    nf_1x1 = [64, 96, 128, 144, 192]
    n_incep = [2, 2, 2, 2, 1]

    group_i = conv1_3
    n_curr_ch = 32
    for i in range(5):
        group_i = pool(group_i, kernel=(2,2)) # 96 48 24 12 6
        for j in range(n_incep[i]):
            group_i, n_curr_ch = inception_group(group_i, 'group{}/unit{}'.format(i+2, j+1), n_curr_ch, 
                    num_filter_3x3=nf_3x3[i], num_filter_1x1=nf_1x1[i],  
                    use_global_stats=is_test, fix_gamma=fix_gamma) 
    return group_i

def get_symbol(num_classes=1000, prefix_symbol='', **kwargs):
    """ PLEEEASEEEEEEEEEEE!!!! """
    is_test = False
    if 'is_test' in kwargs:
        is_test = kwargs['is_test']
    data = mx.sym.Variable('data') # 200

    group6 = get_hjnet_conv(data, is_test, fix_gamma=False)
    pool6 = pool(group6, kernel=(2,2)) 
    conv7 = bn_relu_conv(pool6, num_filter=256, postfix_name='7', 
            kernel=(1,1), use_global_stats=is_test, fix_gamma=False) # fc layer
    conv8 = bn_relu_conv(conv7, num_filter=num_classes, postfix_name='8', 
            kernel=(3,3), use_global_stats=is_test, fix_gamma=False, no_bias=False) # fc layer
    flat = mx.sym.Flatten(data=conv8)
    return mx.sym.SoftmaxOutput(data=flat, name='softmax')

if __name__ == '__main__':
    import os
    os.environ['MXNET_ENGINE_TYPE'] = 'NaiveEngine'
    net, arg_params, aux_params = mx.model.load_checkpoint('../model/imagenet1k-hjnet_preact', 4)
    net = get_symbol(1000)

    mod = mx.mod.Module(net)
    mod.bind(data_shapes=[('data', (1, 3, 200, 200))])
    mod.set_params(arg_params, aux_params, allow_missing=True)
    import ipdb
    ipdb.set_trace()

    syms = mod.symbol.get_internals()
    _, out_shapes, _ = syms.infer_shape(**{'data': (8, 3, 200, 200)})

    for lname, lshape in zip(syms.list_outputs(), out_shapes):
        if lname.endswith('_output'):
            print '%s: %s' % (lname, str(lshape))

    # _, arg_params, aux_params = mx.model.load_checkpoint('../model/imagenet1k-hjnet_preact', 3)
    # for lname in sorted(arg_params):
    #     print '%s: %s' % (lname, str(arg_params[lname].shape))

    import ipdb
    ipdb.set_trace()

    print 'done'
