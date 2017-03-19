import mxnet as mx

def conv_bn_relu(data, group_name, num_filter, kernel, pad, stride, use_global):
    """ used in internal inception layers """
    bn_name = group_name + '_bn'
    conv_name = group_name + '_conv'

    conv = mx.symbol.Convolution(data, name=conv_name, 
            num_filter=num_filter, pad=pad, kernel=kernel, stride=stride, no_bias=True)
    bn = mx.symbol.BatchNorm(conv, name=bn_name, use_global_stats=use_global, fix_gamma=False)
    relu = mx.symbol.Activation(bn, act_type='relu')
    return relu

def bn_relu_conv(data, group_name, num_filter, kernel, pad, stride, use_global):
    """ used in mCReLU """
    bn_name = group_name + '_bn'
    conv_name = group_name + '_conv'

    bn = mx.symbol.BatchNorm(data, name=bn_name, use_global_stats=use_global, fix_gamma=False)
    relu = mx.symbol.Activation(bn, act_type='relu')
    conv = mx.symbol.Convolution(relu, name=conv_name, 
            num_filter=num_filter, pad=pad, kernel=kernel, stride=stride, no_bias=False)
    return conv

def bn_crelu_conv(data, group_name, num_filter, kernel, pad, stride, use_global):
    """ used in mCReLU """
    concat_name = group_name + '_concat'
    bn_name = group_name + '_bn'
    conv_name = group_name + '_conv'

    concat = mx.symbol.concat(data, -data, name=concat_name)
    bn = mx.symbol.BatchNorm(concat, name=bn_name, use_global_stats=use_global, fix_gamma=False)
    relu = mx.symbol.Activation(data, act_type='relu')
    conv = mx.symbol.Convolution(relu, name=conv_name, 
            num_filter=num_filter, pad=pad, kernel=kernel, stride=stride, no_bias=False)
    return conv

def mCReLU(data, group_name, filters, strides, use_global, n_curr_ch):
    """ 
    """
    kernels = ((1,1), (3,3), (1,1))
    pads = ((0,0), (1,1), (0,0))

    conv1 = bn_relu_conv(data, group_name=group_name+'_1', 
            num_filter=filters[0], pad=pads[0], kernel=kernels[0], stride=strides[0], use_global=use_global)
    conv2 = bn_relu_conv(conv1, group_name=group_name+'_2', 
            num_filter=filters[1], pad=pads[1], kernel=kernels[1], stride=strides[1], use_global=use_global)
    conv3 = bn_crelu_conv(conv2, group_name=group_name+'_3', 
            num_filter=filters[2], pad=pads[2], kernel=kernels[2], stride=strides[2], use_global=use_global)

    ss = 1
    for s in strides:
        ss *= s[0]
    need_proj = (n_curr_ch != filters[2]) or (ss != 1)
    if need_proj:
        proj = mx.symbol.Convolution(data, name=group_name+'_proj', 
                num_filter=filters[2], pad=(0,0), kernel=(1,1), stride=(ss,ss))
        res = conv3 + proj
    else:
        res = conv3 + data
    return res, filters[2]

# final_bn is to handle stupid redundancy in the original model
def inception(data, group_name, 
        filter_1, filters_3, filters_5, filter_p, filter_out, stride, use_global, n_curr_ch, final_bn=False):
    """
    """
    group_name = group_name + '_incep'

    group_name_0 = group_name + '_0'
    group_name_1 = group_name + '_1'
    group_name_2 = group_name + '_2'

    incep_bn = mx.symbol.BatchNorm(data, name=group_name+'_bn', use_global_stats=use_global, fix_gamma=False)
    incep_relu = mx.symbol.Activation(incep_bn, act_type='relu')

    incep_0 = conv_bn_relu(incep_relu, group_name=group_name_0, 
            num_filter=filter_1, kernel=(1,1), pad=(0,0), stride=stride, use_global=use_global)

    incep_1_reduce = conv_bn_relu(incep_relu, group_name=group_name_1+'_reduce', 
            num_filter=filters_3[0], kernel=(1,1), pad=(0,0), stride=stride, use_global=use_global)
    incep_1_0 = conv_bn_relu(incep_1_reduce, group_name=group_name_1+'_0', 
            num_filter=filters_3[1], kernel=(3,3), pad=(1,1), stride=(1,1), use_global=use_global)

    incep_2_reduce = conv_bn_relu(incep_relu, group_name=group_name_2+'_reduce', 
            num_filter=filters_5[0], kernel=(1,1), pad=(0,0), stride=stride, use_global=use_global)
    incep_2_0 = conv_bn_relu(incep_2_reduce, group_name=group_name_2+'_0', 
            num_filter=filters_5[1], kernel=(3,3), pad=(1,1), stride=(1,1), use_global=use_global)
    incep_2_1 = conv_bn_relu(incep_2_0, group_name=group_name_2+'_1', 
            num_filter=filters_5[2], kernel=(3,3), pad=(1,1), stride=(1,1), use_global=use_global)

    incep_layers = [incep_0, incep_1_0, incep_2_1]

    if filter_p is not None:
        incep_p_pool = mx.symbol.Pooling(incep_relu, pooling_convention='full', 
                pad=(0,0), kernel=(3,3), stride=(2,2), pool_type='max')
        incep_p_proj = conv_bn_relu(incep_p_pool, group_name=group_name+'_poolproj', 
                num_filter=filter_p, kernel=(1,1), pad=(0,0), stride=(1,1), use_global=use_global)
        incep_layers.append(incep_p_proj)

    incep = mx.symbol.concat(*incep_layers, name=group_name)
    out_conv = mx.symbol.Convolution(incep, name=group_name.replace('_incep', '_out_conv'), 
            num_filter=filter_out, kernel=(1,1), stride=(1,1), pad=(0,0))

    # final_bn is to handle stupid redundancy in the original model
    if final_bn:
        out_conv = mx.symbol.BatchNorm(out_conv, name=group_name.replace('_incep', '_out_bn'), 
                use_global_stats=use_global, fix_gamma=False)
    
    if n_curr_ch != filter_out or stride[0] > 1:
        out_proj = mx.symbol.Convolution(data, name=group_name.replace('_incep', '_proj'), 
                num_filter=filter_out, kernel=(1,1), stride=stride, pad=(0,0))
        res = out_conv + out_proj
    else:
        res = out_conv + data
    return res, filter_out

def pvanet_preact(num_classes, is_test=False):
    """ PVANet 9.0 """
    data = mx.symbol.Variable(name='data')
    conv1_1_conv = mx.symbol.Convolution(data, name='conv1_1_conv', 
            num_filter=16, pad=(0,0), kernel=(7,7), stride=(2,2), no_bias=True)
    conv1_1_concat = mx.symbol.concat(conv1_1_conv, -conv1_1_conv, name='conv1_1_concat')
    conv1_1_bn = mx.symbol.BatchNorm(conv1_1_concat, name='conv1_1_bn', 
            use_global_stats=is_test, fix_gamma=False)
    conv1_1_relu = mx.symbol.Activation(conv1_1_bn, act_type='relu')
    pool1 = mx.symbol.Pooling(conv1_1_relu, kernel=(3,3), stride=(2,2), pool_type='max', 
            pooling_convention='full')
    
    # no pre bn-scale-relu for 2_1_1
    conv2_1_1_conv = mx.symbol.Convolution(pool1, name='conv2_1_1_conv', 
            num_filter=24, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
    conv2_1_2_conv = bn_relu_conv(conv2_1_1_conv, group_name='conv2_1_2', 
            num_filter=24, kernel=(3,3), pad=(1,1), stride=(1,1), use_global=is_test)
    conv2_1_3_conv = bn_crelu_conv(conv2_1_2_conv, group_name='conv2_1_3', 
            num_filter=64, kernel=(1,1), pad=(0,0), stride=(1,1), use_global=is_test)
    conv2_1_proj = mx.symbol.Convolution(pool1, name='conv2_1_proj', 
            num_filter=64, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
    conv2_1 = conv2_1_3_conv + conv2_1_proj

    # stack up mCReLU layers
    n_curr_ch = 64
    conv2_2, n_curr_ch = mCReLU(conv2_1, group_name='conv2_2', 
            filters=(24, 24, 64), strides=((1,1),(1,1),(1,1)), use_global=is_test, n_curr_ch=n_curr_ch)
    conv2_3, n_curr_ch = mCReLU(conv2_2, group_name='conv2_3', 
            filters=(24, 24, 64), strides=((1,1),(1,1),(1,1)), use_global=is_test, n_curr_ch=n_curr_ch)
    conv3_1, n_curr_ch = mCReLU(conv2_3, group_name='conv3_1', 
            filters=(48, 48, 128), strides=((2,2),(1,1),(1,1)), use_global=is_test, n_curr_ch=n_curr_ch)
    conv3_2, n_curr_ch = mCReLU(conv3_1, group_name='conv3_2', 
            filters=(48, 48, 128), strides=((1,1),(1,1),(1,1)), use_global=is_test, n_curr_ch=n_curr_ch)
    conv3_3, n_curr_ch = mCReLU(conv3_2, group_name='conv3_3', 
            filters=(48, 48, 128), strides=((1,1),(1,1),(1,1)), use_global=is_test, n_curr_ch=n_curr_ch)
    conv3_4, n_curr_ch = mCReLU(conv3_3, group_name='conv3_4', 
            filters=(48, 48, 128), strides=((1,1),(1,1),(1,1)), use_global=is_test, n_curr_ch=n_curr_ch)

    # stack up inception layers
    conv4_1, n_curr_ch = inception(conv3_4, group_name='conv4_1', 
            filter_1=64, filters_3=(48,128), filters_5=(24,48,48), filter_p=128, filter_out=256, 
            stride=(2,2), use_global=is_test, n_curr_ch=n_curr_ch)
    conv4_2, n_curr_ch = inception(conv4_1, group_name='conv4_2', 
            filter_1=64, filters_3=(64,128), filters_5=(24,48,48), filter_p=None, filter_out=256, 
            stride=(1,1), use_global=is_test, n_curr_ch=n_curr_ch)
    conv4_3, n_curr_ch = inception(conv4_2, group_name='conv4_3', 
            filter_1=64, filters_3=(64,128), filters_5=(24,48,48), filter_p=None, filter_out=256, 
            stride=(1,1), use_global=is_test, n_curr_ch=n_curr_ch)
    conv4_4, n_curr_ch = inception(conv4_3, group_name='conv4_4', 
            filter_1=64, filters_3=(64,128), filters_5=(24,48,48), filter_p=None, filter_out=256, 
            stride=(1,1), use_global=is_test, n_curr_ch=n_curr_ch)
    conv5_1, n_curr_ch = inception(conv4_4, group_name='conv5_1', 
            filter_1=64, filters_3=(96,192), filters_5=(32,64,64), filter_p=128, filter_out=384, 
            stride=(2,2), use_global=is_test, n_curr_ch=n_curr_ch)
    conv5_2, n_curr_ch = inception(conv5_1, group_name='conv5_2', 
            filter_1=64, filters_3=(96,192), filters_5=(32,64,64), filter_p=None, filter_out=384, 
            stride=(1,1), use_global=is_test, n_curr_ch=n_curr_ch)
    conv5_3, n_curr_ch = inception(conv5_2, group_name='conv5_3', 
            filter_1=64, filters_3=(96,192), filters_5=(32,64,64), filter_p=None, filter_out=384, 
            stride=(1,1), use_global=is_test, n_curr_ch=n_curr_ch)
    conv5_4, n_curr_ch = inception(conv5_3, group_name='conv5_4', 
            filter_1=64, filters_3=(96,192), filters_5=(32,64,64), filter_p=None, filter_out=384, 
            stride=(1,1), use_global=is_test, n_curr_ch=n_curr_ch, final_bn=True)

    # final layers
    conv5_4_last_bn = mx.symbol.BatchNorm(conv5_4, name='conv5_4_last_bn', use_global_stats=is_test)
    conv5_4_last_relu = mx.symbol.Activation(conv5_4_last_bn, act_type='relu')

    # return conv5_4_last_relu

    # I don't think we need this
    # pool5 = mx.symbol.Pooling(name='pool5', data=conv5_4_last_relu, pooling_convention='full', 
    #         pad=(0,0), kernel=(1,1), stride=(1,1), pool_type='max')
    # flatten_0=mx.symbol.Flatten(name='flatten_0', data=pool5)
    flat5 = mx.symbol.Flatten(conv5_4_last_relu)
    fc6 = mx.symbol.FullyConnected(flat5, name='fc6', num_hidden=4096)
    fc6_bn = mx.symbol.BatchNorm(fc6, name='fc6_bn', use_global_stats=is_test, fix_gamma=False)
    # fc6_dropout = mx.symbol.Dropout(name='fc6_dropout', data=fc6_bn, p=0.5)
    fc6_relu = mx.symbol.Activation(fc6_bn, act_type='relu')
    fc7 = mx.symbol.FullyConnected(fc6_relu, name='fc7', num_hidden=4096)
    fc7_bn = mx.symbol.BatchNorm(fc7, name='fc7_bn', use_global_stats=is_test, fix_gamma=False)
    # fc7_dropout = mx.symbol.Dropout(name='fc7_dropout', data=fc7_bn, p=0.5)
    fc7_relu = mx.symbol.Activation(fc7_bn, act_type='relu')
    fc8 = mx.symbol.FullyConnected(fc7_relu, name='fc8', num_hidden=num_classes)

    return fc8

def get_symbol(num_classes, **kwargs):
    """
    """
    return pvanet_preact(num_classes, is_test=True)
    fc8 = pvanet_preact(num_classes, is_test=True)
    softmax = mx.symbol.SoftmaxOutput(data=fc8, name='softmax')

    return softmax


if __name__ == '__main__':
    net = get_symbol(1000)

    mod = mx.mod.Module(net)
    mod.bind(data_shapes=[('data', (1, 3, 230, 230))])

    mod.init_params()
    # arg_params, aux_params = mod.get_params()
    # for k in sorted(arg_params):
    #     print k + ': ' + str(arg_params[k].shape)

    syms = mod.symbol.get_internals()
    _, out_shapes, _ = syms.infer_shape(**{'data': (8, 3, 230, 230)})

    for lname, lshape in zip(syms.list_outputs(), out_shapes):
        if lname.endswith('_output'):
            print '%s: %s' % (lname, str(lshape))

    import ipdb
    ipdb.set_trace()

    print 'done'
