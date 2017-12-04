import mxnet as mx


def conv_bn_relu(data, name, \
        num_filter, kernel=(3, 3), pad=(1, 1), stride=(1, 1), num_group=1, wd_mult=1.0, \
        use_global_stats=False):
    #
    conv_name='conv'+name
    conv_w = mx.sym.var(name=conv_name+'_weight', lr_mult=1.0, wd_mult=wd_mult)
    conv_b = None

    # conv = mx.sym.Convolution(data, name=conv_name, num_filter=num_filter, \
    #         kernel=kernel, pad=pad, stride=stride, num_group=num_group, no_bias=True)
    conv = mx.sym.Convolution(data, name=conv_name, weight=conv_w, bias=conv_b, num_filter=num_filter, \
            kernel=kernel, pad=pad, stride=stride, num_group=num_group, no_bias=True)
    bn = mx.sym.BatchNorm(conv, name=conv_name+'_bn', \
            use_global_stats=use_global_stats, fix_gamma=False, eps=1e-04)
    relu = mx.sym.Activation(bn, name='relu'+name, act_type='relu')
    return relu


def depthwise_unit(data, name, nf_dw, nf_sep, stride=(1, 1), use_global_stats=False):
    #
    conv_dw = conv_bn_relu(data, name=name+'_dw', \
            num_filter=nf_dw, kernel=(3, 3), pad=(1, 1), stride=stride, num_group=1, wd_mult=0.01, \
            use_global_stats=use_global_stats)
    conv_sep = conv_bn_relu(conv_dw, name=name+'_sep', \
            num_filter=nf_sep, kernel=(1, 1), pad=(0, 0), num_group=1, \
            use_global_stats=use_global_stats)
    return conv_sep


def get_symbol(num_classes=1000, **kwargs):
    #
    if 'use_global_stats' not in kwargs:
        use_global_stats = False
    else:
        use_global_stats = kwargs['use_global_stats']

    data = mx.sym.var('data')

    conv1 = conv_bn_relu(data, '1',
            num_filter=32, kernel=(3, 3), pad=(1, 1), stride=(2, 2),
            use_global_stats=use_global_stats)

    nf_dw_all = [(32, 64), (128, 128), (256, 256), (512, 512, 512, 512, 512, 512)]
    nf_sep_all = [(64, 128), (128, 256), (256, 512), (512, 512, 512, 512, 512, 1024)]
    stride_all = [(1, 2), (1, 2), (1, 2), (1, 1, 1, 1, 1, 2)]

    convi = conv1
    for i, (nf_dw_i, nf_sep_i, stride_i) in enumerate(zip(nf_dw_all, nf_sep_all, stride_all), 2):
        for j, (nf_dw, nf_sep, stride) in enumerate(zip(nf_dw_i, nf_sep_i, stride_i), 1):
            print 'nf_dw = {}'.format(nf_dw)
            print 'nf_sep = {}'.format(nf_sep)
            print 'stride = {}'.format(stride)
            name = '{}_{}'.format(i, j)
            ss = (stride, stride)
            convi = depthwise_unit(convi, name,
                    nf_dw=nf_dw, nf_sep=nf_sep, stride=ss, use_global_stats=use_global_stats)

    conv6 = depthwise_unit(convi, '6', nf_dw=128, nf_sep=1024, use_global_stats=use_global_stats)
    pool6 = mx.sym.Pooling(conv6, name='pool6', kernel=(1, 1),
            global_pool=True, pool_type='avg', pooling_convention='full')

    fc7 = mx.sym.Convolution(pool6, name='fc7',
            num_filter=num_classes, kernel=(1, 1), pad=(0, 0), no_bias=False)
    flatten = mx.sym.flatten(fc7, name='flatten')

    softmax = mx.sym.SoftmaxOutput(flatten, name='softmax')
    return softmax
