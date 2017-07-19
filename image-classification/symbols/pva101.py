import mxnet as mx


def conv_bn_relu(data, group_name, 
        num_filter, kernel, pad, stride, no_bias, 
        use_global_stats=True, use_crelu=False, 
        get_syms=False):
    ''' use in mCReLU '''
    conv_name = group_name + ''
    concat_name = group_name + '/concat'
    bn_name = group_name + '/bn'
    relu_name = group_name + '/relu'

    syms = {}
    conv = mx.sym.Convolution(data, name=conv_name, 
            num_filter=num_filter, kernel=kernel, pad=pad, stride=stride, no_bias=no_bias)
    syms['conv'] = conv
    if use_crelu:
        conv = mx.sym.concat(conv, -conv, name=concat_name)
        syms['concat'] = conv
    bn = mx.sym.BatchNorm(conv, name=bn_name, use_global_stats=use_global_stats, fix_gamma=False)
    syms['bn'] = bn
    relu = mx.sym.Activation(bn, name=relu_name, act_type='relu')
    if get_syms:
        return relu, syms
    else:
        return relu


def mcrelu(data, prefix_group, filters, no_bias, use_global_stats):
    ''' conv2 and conv3 '''
    group1 = conv_bn_relu(data, group_name=prefix_group+'_1', 
            num_filter=filters[0], kernel=(1,1), pad=(0,0), stride=(1,1), no_bias=no_bias,
            use_global_stats=use_global_stats)
    group2 = conv_bn_relu(group1, group_name=prefix_group+'_2',
            num_filter=filters[1], kernel=(3,3), pad=(1,1), stride=(2,2), no_bias=no_bias,
            use_global_stats=use_global_stats, use_crelu=True)
    conv3 = mx.sym.Convolution(group2, name=prefix_group+'_3/out',
            num_filter=filters[2], kernel=(1,1), pad=(0,0), stride=(1,1), no_bias=no_bias)
    proj1 = mx.sym.Convolution(data, name=prefix_group+'_1/proj',
            num_filter=filters[2], kernel=(1,1), pad=(0,0), stride=(2,2), no_bias=no_bias)
    bn3 = mx.sym.BatchNorm(conv3+proj1, name=prefix_group+'_3/elt/bn', 
            use_global_stats=use_global_stats, fix_gamma=False)
    relu3 = mx.sym.Activation(bn3, name=prefix_group+'_3/relu', act_type='relu')
    return relu3


def inception(data, prefix_group, 
        filters_1, filters_3, filters_5, no_bias, 
        use_global_stats, do_pool=False):
    ''' inception group '''
    if do_pool:
        pool1 = mx.sym.Pooling(data, name=prefix_group+'/pool1', kernel=(3,3), pad=(0,0), stride=(2,2), 
                pool_type='max', pooling_convention='full')
        ss = (2, 2)
    else:
        pool1 = data
        ss = (1, 1)
    # conv1
    conv1 = conv_bn_relu(pool1, group_name=prefix_group+'/conv1', 
            num_filter=filters_1, kernel=(1,1), pad=(0,0), stride=(1,1), no_bias=no_bias,
            use_global_stats=use_global_stats)
    # conv3
    conv3_1 = conv_bn_relu(data, group_name=prefix_group+'/conv3_1', 
            num_filter=filters_3[0], kernel=(1,1), pad=(0,0), stride=(1,1), no_bias=no_bias,
            use_global_stats=use_global_stats)
    conv3_2 = conv_bn_relu(conv3_1, group_name=prefix_group+'/conv3_2', 
            num_filter=filters_3[1], kernel=(3,3), pad=(1,1), stride=ss, no_bias=no_bias,
            use_global_stats=use_global_stats)
    # conv5
    conv5_1 = conv_bn_relu(data, group_name=prefix_group+'/conv5_1', 
            num_filter=filters_5[0], kernel=(1,1), pad=(0,0), stride=(1,1), no_bias=no_bias,
            use_global_stats=use_global_stats)
    conv5_2 = conv_bn_relu(conv5_1, group_name=prefix_group+'/conv5_2', 
            num_filter=filters_5[1], kernel=(3,3), pad=(1,1), stride=(1,1), no_bias=no_bias,
            use_global_stats=use_global_stats)
    conv5_3 = conv_bn_relu(conv5_2, group_name=prefix_group+'/conv5_3', 
            num_filter=filters_5[2], kernel=(3,3), pad=(1,1), stride=ss, no_bias=no_bias,
            use_global_stats=use_global_stats)
    return mx.sym.concat(conv1, conv3_2, conv5_3)


def residual_inc(lhs, rhs, prefix_lhs, prefix_rhs, num_filter, stride, no_bias, use_global_stats):
    ''' residual connection between inception layers '''
    lhs = mx.sym.Convolution(lhs, name=prefix_lhs+'/proj',
            num_filter=num_filter, kernel=(1,1), pad=(0,0), stride=stride, no_bias=no_bias)
    rhs = mx.sym.Convolution(rhs, name=prefix_rhs+'/out',
            num_filter=num_filter, kernel=(1,1), pad=(0,0), stride=(1,1), no_bias=no_bias)
    elt = lhs+rhs
    bn = mx.sym.BatchNorm(elt, name=prefix_rhs+'/elt/bn',
            use_global_stats=use_global_stats, fix_gamma=False)
    relu = mx.sym.Activation(bn, act_type='relu')
    return relu
    # return relu, elt


def pvanet_preact(data, use_global_stats=True, no_bias=False):
    ''' pvanet 10.1 '''
    conv1 = conv_bn_relu(data / 128.0, group_name='conv1', 
            num_filter=12, kernel=(4,4), pad=(1,1), stride=(2,2), no_bias=no_bias, 
            use_global_stats=use_global_stats, use_crelu=True)
    # conv2
    conv2 = mcrelu(conv1, prefix_group='conv2', 
            filters=(12, 18, 36), no_bias=no_bias, use_global_stats=use_global_stats)
    # conv3
    conv3 = mcrelu(conv2, prefix_group='conv3',
            filters=(18, 36, 72), no_bias=no_bias, use_global_stats=use_global_stats)
    # inc3a
    inc3a = inception(conv3, prefix_group='inc3a',
            filters_1=96, filters_3=(16,64), filters_5=(16,32,32), no_bias=no_bias,
            use_global_stats=use_global_stats, do_pool=True)
    # inc3b
    inc3b = inception(inc3a, prefix_group='inc3b',
            filters_1=96, filters_3=(16,64), filters_5=(16,32,32), no_bias=no_bias,
            use_global_stats=use_global_stats)
    # inc3b/residual
    inc3b = residual_inc(conv3, inc3b, prefix_lhs='inc3a', prefix_rhs='inc3b', 
            num_filter=128, stride=(2,2), no_bias=no_bias, use_global_stats=use_global_stats)
    # inc3c
    inc3c = inception(inc3b, prefix_group='inc3c',
            filters_1=96, filters_3=(16,64), filters_5=(16,32,32), no_bias=no_bias,
            use_global_stats=use_global_stats)
    # inc3d
    inc3d = inception(inc3c, prefix_group='inc3d',
            filters_1=96, filters_3=(16,64), filters_5=(16,32,32), no_bias=no_bias,
            use_global_stats=use_global_stats)
    # inc3e
    inc3e = inception(inc3d, prefix_group='inc3e',
            filters_1=96, filters_3=(16,64), filters_5=(16,32,32), no_bias=no_bias,
            use_global_stats=use_global_stats)
    # inc3e/residual
    inc3e = residual_inc(inc3b, inc3e, prefix_lhs='inc3c', prefix_rhs='inc3e', 
            num_filter=128, stride=(1,1), no_bias=no_bias, use_global_stats=use_global_stats)
    # inc4a
    inc4a = inception(inc3e, prefix_group='inc4a',
            filters_1=128, filters_3=(32,96), filters_5=(16,32,32), no_bias=no_bias,
            use_global_stats=use_global_stats, do_pool=True)
    # inc4b
    inc4b = inception(inc4a, prefix_group='inc4b',
            filters_1=128, filters_3=(32,96), filters_5=(16,32,32), no_bias=no_bias,
            use_global_stats=use_global_stats)
    # inc4b/residual
    inc4b = residual_inc(inc3e, inc4b, prefix_lhs='inc4a', prefix_rhs='inc4b', 
            num_filter=192, stride=(2,2), no_bias=no_bias, use_global_stats=use_global_stats)
    # inc4c
    inc4c = inception(inc4b, prefix_group='inc4c',
            filters_1=128, filters_3=(32,96), filters_5=(16,32,32), no_bias=no_bias,
            use_global_stats=use_global_stats)
    # inc4d
    inc4d = inception(inc4c, prefix_group='inc4d',
            filters_1=128, filters_3=(32,96), filters_5=(16,32,32), no_bias=no_bias,
            use_global_stats=use_global_stats)
    # inc4e
    inc4e = inception(inc4d, prefix_group='inc4e',
            filters_1=128, filters_3=(32,96), filters_5=(16,32,32), no_bias=no_bias,
            use_global_stats=use_global_stats)
    # inc4e/residual
    inc4e = residual_inc(inc4b, inc4e, prefix_lhs='inc4c', prefix_rhs='inc4e', 
            num_filter=384, stride=(1,1), no_bias=no_bias, use_global_stats=use_global_stats)

    return inc4e


def get_symbol(num_classes=1278, image_shape=(192, 192), **kwargs):
    ''' network for training '''
    # global parameters, (maybe will be changed)
    no_bias = False
    use_global_stats = False

    data = mx.sym.Variable(name='data')

    # shared conv layers
    inc4e = pvanet_preact(data, no_bias=no_bias, use_global_stats=use_global_stats)
    pool5 = mx.sym.Pooling(inc4e, kernel=(3, 3), stride=(2, 2), pool_type='max')
    flat5 = mx.symbol.Flatten(name='flat5', data=pool5)
    fc6 = mx.symbol.FullyConnected(name='fc6', data=flat5, num_hidden=4096)
    fc6_bn = mx.symbol.BatchNorm(name='fc6_bn', data=fc6, use_global_stats=use_global_stats, fix_gamma=False)
    fc6_dropout = mx.symbol.Dropout(name='fc6_dropout', data=fc6_bn, p=0.5)
    fc6_relu = mx.symbol.Activation(name='fc6_relu', data=fc6_dropout, act_type='relu')
    fc7 = mx.symbol.FullyConnected(name='fc7', data=fc6_relu, num_hidden=4096)
    fc7_bn = mx.symbol.BatchNorm(name='fc7_bn', data=fc7, use_global_stats=use_global_stats, fix_gamma=False)
    fc7_dropout = mx.symbol.Dropout(name='fc7_dropout', data=fc7_bn, p=0.5)
    fc7_relu = mx.symbol.Activation(name='fc7_relu', data=fc7_dropout, act_type='relu')
    fc8 = mx.symbol.FullyConnected(name='fc8', data=fc7_relu, num_hidden=num_classes)

    softmax = mx.sym.SoftmaxOutput(data=fc8, name='softmax')
    return softmax
