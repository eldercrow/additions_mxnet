import mxnet as mx


def bn_relu_conv(data, prefix_name, num_filter,
                 kernel=(3, 3), pad=(1, 1), stride=(1, 1), num_group=1,
                 wd_mult=1.0, no_bias=True,
                 use_global_stats=False):
    #
    assert prefix_name != ''
    conv_name = prefix_name + 'conv'
    bn_name = prefix_name + 'bn'

    bn = mx.sym.BatchNorm(data, name=bn_name,
            use_global_stats=use_global_stats, fix_gamma=False, eps=1e-04, momentum=0.99)

    # relu = bn * mx.sym.Activation(bn, act_type='sigmoid')
    relu = mx.sym.Activation(bn, act_type='relu')

    conv_w = mx.sym.var(name=conv_name+'_weight', lr_mult=1.0, wd_mult=wd_mult)
    conv = mx.sym.Convolution(relu, name=conv_name, weight=conv_w,
            num_filter=num_filter, kernel=kernel, pad=pad, stride=stride, num_group=num_group,
            no_bias=no_bias)

    return conv


def bn_conv(data, prefix_name,
            num_filter, kernel=(1, 1), pad=(0, 0),
            no_bias=True,
            use_global_stats=False):
    #
    assert prefix_name != ''
    conv_name = prefix_name + 'conv'
    bn_name = prefix_name + 'bn'

    bn = mx.sym.BatchNorm(data, name=bn_name,
            use_global_stats=use_global_stats, fix_gamma=False, eps=1e-04, momentum=0.99)

    conv = mx.sym.Convolution(bn, name=conv_name,
            num_filter=num_filter, kernel=kernel, pad=pad,
            no_bias=no_bias)

    return conv


def depthwise_conv(data, name, nf_dw, nf_sep, maxout_ratio=1,
        kernel=(3, 3), pad=(1, 1), stride=(1, 1),
        use_global_stats=False):
    #
    conv3x3 = bn_relu_conv(data, name+'3x3/',
            num_filter=nf_dw, kernel=kernel, pad=pad, stride=stride, num_group=nf_dw,
            wd_mult=0.0, use_global_stats=use_global_stats)

    if nf_sep == 0:
        conv1x1 = conv3x3
    else:
        conv1x1 = bn_conv(conv3x3, name+'1x1/',
                num_filter=nf_sep,
                use_global_stats=use_global_stats)

    return conv1x1


def get_symbol(num_classes=1000, **kwargs):
    '''
    '''
    use_global_stats = kwargs['use_global_stats']

    data = mx.symbol.Variable(name="data")
    label = mx.symbol.Variable(name="label")

    conv1 = mx.sym.Convolution(data, name='1/conv',
            num_filter=32, kernel=(4, 4), pad=(1, 1), stride=(2, 2), no_bias=True)  # 32, 198
    concat1 = mx.sym.concat(conv1, -conv1, name='1/concat')
    bn1 = mx.sym.BatchNorm(concat1, name='1/bn', fix_gamma=False,
            eps=1e-04, momentum=0.99, use_global_stats=use_global_stats)

    conv2 = mx.sym.Pooling(bn1, name='2/', kernel=(4, 4), pad=(1, 1), stride=(2, 2), pool_type='max')

    conv3_1 = depthwise_conv(conv2, '3_1/',
            nf_dw=64, nf_sep=64, kernel=(5, 5), pad=(2, 2),
            use_global_stats=use_global_stats)
    conv3_2 = mx.sym.concat(conv3_1, -conv3_1, name='3_2/concat')
    # conv3_2 = depthwise_conv(conv3_1, '3_2/',
    #         nf_dw=64, nf_sep=128, maxout_ratio=2, kernel=(3, 3), pad=(1, 1),
    #         use_global_stats=use_global_stats)

    nf_dw_all = [256, 512, 1024]
    nf_sep_all = [256, 512, 1024]
    n_unit_all = [2, 4, 2]

    groups = []
    g = conv3_2

    for i, (nf_dw, nf_sep, nu) in enumerate(zip(nf_dw_all, nf_sep_all, n_unit_all)):
        #
        for j in range(nu):
            if j == 0:
                p = mx.sym.Pooling(g, name='g{}/pool/'.format(i),
                        kernel=(4, 4), pad=(1, 1), stride=(2, 2), pool_type='max')
                g = depthwise_conv(g, 'g{}/dn/'.format(i, j),
                        nf_dw=nf_dw/2, nf_sep=0,
                        kernel=(4, 4), pad=(1, 1), stride=(2, 2),
                        use_global_stats=use_global_stats)
                g = mx.sym.concat(g, p)
                g = bn_conv(g, 'g{}/u{}/'.format(i, j),
                        num_filter=nf_sep, kernel=(1, 1), pad=(0, 0),
                        use_global_stats=use_global_stats)
            else:
                g1 = depthwise_conv(g, 'g{}/u{}/'.format(i, j),
                        nf_dw=nf_dw, nf_sep=nf_sep,
                        kernel=(5, 5), pad=(2, 2),
                        use_global_stats=use_global_stats)
                g = mx.sym.broadcast_add(g, g1, name='g{}/u{}'.format(i, j))
        groups.append(g)

    # hyper_groups = []
    # nf_hyper = [(256, 128) for _ in groups]
    #
    # nu_cls = (3, 3, 3, 2, 1, 1)
    # for i, (g, nf) in enumerate(zip(groups, nf_hyper)):
    #     # classification feature
    #     p1 = relu_conv_bn(g, 'hypercls{}/init/'.format(i),
    #             num_filter=nf[0], kernel=(1, 1), pad=(0, 0),
    #             use_global_stats=use_global_stats)
    #     for j in range(nu_cls[i]):
    #         p1 = depthwise_conv(p1, 'hypercls{}/{}/'.format(i, j),
    #                 num_filter=nf[0], kernel=(3, 3), pad=(1, 1),
    #                 use_global_stats=use_global_stats)
    #     h1 = mx.sym.Activation(p1, name='hyper{}/1'.format(i), act_type='relu')
    #
    #     # regression feature
    #     p2 = relu_conv_bn(g, 'hyperreg{}/init/'.format(i),
    #             num_filter=nf[1], kernel=(1, 1), pad=(0, 0),
    #             use_global_stats=use_global_stats)
    #     p2 = depthwise_conv(p2, 'hyperreg{}/0/'.format(i),
    #             num_filter=nf[1], kernel=(3, 3), pad=(1, 1),
    #             use_global_stats=use_global_stats)
    #     h2 = mx.sym.Activation(p2, name='hyper{}/2'.format(i), act_type='relu')
    #
    #     hyper_groups.append((h1, h2))
    #
    # pooled = []
    # ps = 8
    # for i, h in enumerate(hyper_groups):
    #     hc = mx.sym.concat(h[0], h[1])
    #     if ps > 1:
    #         p = mx.sym.Pooling(hc, kernel=(ps, ps), stride=(ps, ps), pool_type='max')
    #     else:
    #         p = hc
    #     ps /= 2
    #     pooled.append(p)
    #
    # pooled_all = mx.sym.flatten(mx.sym.concat(*pooled), name='flatten')
    # fc1 = mx.sym.FullyConnected(pooled_all, num_hidden=4096, name='fc1', no_bias=True)
    # bn_fc1 = mx.sym.BatchNorm(fc1, use_global_stats=use_global_stats, fix_gamma=False)
    # relu_fc1 = mx.sym.Activation(bn_fc1, act_type='relu')
    # fc2 = mx.sym.FullyConnected(relu_fc1, num_hidden=num_classes, name='fc2')
    # cls_prob = mx.sym.softmax(fc2, name='cls_prob')
    # softmax = mx.sym.Custom(fc2, cls_prob, label, op_type='smoothed_softmax_loss', name='softmax',
    #         th_prob=1e-05, normalization='null')
    # # softmax = mx.sym.SoftmaxOutput(data=fc2, label=label, name='softmax')
    # return softmax

    bn6 = mx.sym.BatchNorm(groups[-1], name='bn6', fix_gamma=False, eps=1e-04, momentum=0.99)
    relu6 = mx.sym.Activation(bn6, name='relu6', act_type='relu')

    # from the original classification network
    pool6 = mx.sym.Pooling(relu6, name='pool6', kernel=(1, 1), global_pool=True, pool_type='avg')
    fc7 = mx.sym.Convolution(pool6, name='fc7',
            num_filter=num_classes, kernel=(1, 1), pad=(0, 0), no_bias=False)
    flatten = mx.sym.flatten(fc7, name='flatten')

    softmax = mx.sym.SoftmaxOutput(flatten, name='softmax')
    return softmax
