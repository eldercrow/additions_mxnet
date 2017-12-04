import mxnet as mx


def pool(data, name=None, kernel=(2, 2), pad=(0, 0), stride=(2, 2), pool_type='max'):
    return mx.sym.Pooling(data, name=name, kernel=kernel, pad=pad, stride=stride, pool_type=pool_type)


def relu_conv_bn(data, prefix_name,
                 num_filter, kernel=(3, 3), pad=(1, 1), stride=(1, 1), num_group=1,
                 wd_mult=1.0, no_bias=True, use_crelu=False,
                 use_global_stats=False):
    #
    assert prefix_name != ''
    conv_name = prefix_name + 'conv'
    bn_name = prefix_name + 'bn'

    relu = mx.sym.Activation(data, act_type='relu')

    conv_w = mx.sym.var(name=conv_name+'_weight', lr_mult=1.0, wd_mult=wd_mult)
    conv = mx.sym.Convolution(relu, name=conv_name, weight=conv_w,
            num_filter=num_filter, kernel=kernel, pad=pad, stride=stride, num_group=num_group,
            no_bias=no_bias)
    if use_crelu:
        conv = mx.sym.concat(conv, -conv)

    bn = mx.sym.BatchNorm(conv, name=bn_name,
            use_global_stats=use_global_stats, fix_gamma=False, eps=1e-04, momentum=0.99)
    return bn


def depthwise_conv(data, name, num_filter,
        kernel=(3, 3), pad=(1, 1), stride=(1, 1),
        use_crelu=False, use_global_stats=False):
    #
    conv_name = name + 'conv'
    bn_name = name + 'bn'

    relu = mx.sym.Activation(data, act_type='relu')

    conv_w = mx.sym.var(name=conv_name+'/3x3_weight', lr_mult=1.0, wd_mult=0.01)
    conv3x3 = mx.sym.Convolution(relu, name=conv_name+'/3x3', weight=conv_w,
            num_filter=num_filter, kernel=kernel, pad=pad, stride=stride, num_group=num_filter)
    conv1x1 = mx.sym.Convolution(conv3x3, name=conv_name+'/1x1',
            num_filter=num_filter, kernel=(1, 1), pad=(0, 0), no_bias=True)

    if use_crelu:
        conv1x1 = mx.sym.concat(conv1x1, -conv1x1)

    bn = mx.sym.BatchNorm(conv1x1, name=bn_name,
            use_global_stats=use_global_stats, fix_gamma=False, eps=1e-04, momentum=0.99)
    return bn


# def subpixel_upsample(data, ch, c, r, name=None):
#     '''
#     Transform input data shape of (n, ch*r*c, h, w) to (n, ch, h*r, c*w).
#
#     ch: number of channels after upsample
#     r: row scale factor
#     c: column scale factor
#     '''
#     if r == 1 and c == 1:
#         return data
#     X = mx.sym.reshape(data=data, shape=(-3, 0, 0))  # (n*ch*r*c, h, w)
#     X = mx.sym.reshape(
#         data=X, shape=(-4, -1, r * c, 0, 0))  # (n*ch, r*c, h, w)
#     X = mx.sym.transpose(data=X, axes=(0, 3, 2, 1))  # (n*ch, w, h, r*c)
#     X = mx.sym.reshape(data=X, shape=(0, 0, -1, c))  # (n*ch, w, h*r, c)
#     X = mx.sym.transpose(data=X, axes=(0, 2, 1, 3))  # (n*ch, h*r, w, c)
#     X = mx.sym.reshape(data=X, name=name, shape=(-4, -1, ch, 0, -3))  # (n, ch, h*r, w*c)
#     return X
#
#
# def proj_add(lhs, rhs, name, num_filter, do_pool, use_global_stats):
#     #
#     lhs = pool(lhs, kernel=(3, 3), pad=(1, 1)) if do_pool else lhs
#     lhs = relu_conv_bn(lhs, name+'lhs/',
#             num_filter=num_filter, kernel=(1, 1), pad=(0, 0),
#             use_global_stats=use_global_stats)
#     rhs = relu_conv_bn(rhs, name+'rhs/',
#             num_filter=num_filter, kernel=(1, 1), pad=(0, 0),
#             use_global_stats=use_global_stats)
#     return mx.sym.broadcast_add(lhs, rhs, name=name+'add/')


def topdown_feature(data, updata, name, scale, nch_up, nf_proj, nf_all, use_global_stats):
    #
    # upsample, proj, concat, mix
    updata = mx.sym.UpSampling(updata, scale=scale, sample_type='bilinear',
            num_filter=nch_up, name=name+'upsample')
    updata = relu_conv_bn(updata, name+'proj/',
            num_filter=nf_proj, kernel=(1, 1), pad=(0, 0),
            use_global_stats=use_global_stats)

    data = mx.sym.concat(data, updata, name=name+'concat')
    data = depthwise_conv(data, name+'mix/',
            num_filter=nf_all, use_global_stats=use_global_stats)
    return data


def prepare_groups(data, use_global_stats):
    ''' prepare basic groups '''
    # 48 24 12 6 3 1
    num_filters = [256, 512, 1024]
    num_units = [2, 4, 2]

    groups = []
    g = data
    for i, (nf, nu) in enumerate(zip(num_filters, num_units)):
        #
        g0 = relu_conv_bn(g, 'g{}/init/'.format(i),
                num_filter=nf, kernel=(1, 1), pad=(0, 0),
                use_global_stats=use_global_stats)
        g0 = pool(g0)

        g = g0
        for j in range(nu):
            g = depthwise_conv(g, 'g{}/u{}/'.format(i, j),
                    num_filter=nf, kernel=(3, 3), pad=(1, 1),
                    use_global_stats=use_global_stats)

        groups.append(g)

    return groups
    #
    # # top-down features
    # groups[0] = topdown_feature(groups[0], groups[2], 'up0/', scale=4,
    #         nch_up=1024, nf_proj=256, nf_all=512, use_global_stats=use_global_stats)
    #
    # groups[2] = depthwise_conv(groups[2], 'g2/dil/',
    #         num_filter=512, use_global_stats=use_global_stats)
    #
    # g = groups[2]
    #
    # g = relu_conv_bn(g, 'g3/init/',
    #         num_filter=512, kernel=(1, 1), pad=(0, 0),
    #         use_global_stats=use_global_stats)
    # g = pool(g)
    # g = depthwise_conv(g, 'g3/0/',
    #         num_filter=512, kernel=(3, 3), pad=(1, 1),
    #         use_global_stats=use_global_stats)
    # groups.append(g)
    #
    # g = relu_conv_bn(g, 'g4/init/',
    #         num_filter=512, kernel=(1, 1), pad=(0, 0),
    #         use_global_stats=use_global_stats)
    # g = pool(g)
    # g = depthwise_conv(g, 'g4/0/',
    #         num_filter=512, kernel=(3, 3), pad=(1, 1),
    #         use_global_stats=use_global_stats)
    # groups.append(g)
    #
    # g = relu_conv_bn(g, 'g5/init/',
    #         num_filter=512, kernel=(1, 1), pad=(0, 0),
    #         use_global_stats=use_global_stats)
    # g = depthwise_conv(g, 'g5/0/',
    #         num_filter=512, kernel=(3, 3), pad=(0, 0),
    #         use_global_stats=use_global_stats)
    # groups.append(g)
    #
    # return groups


def get_symbol(num_classes=1000, **kwargs):
    '''
    '''
    use_global_stats = kwargs['use_global_stats']

    data = mx.symbol.Variable(name="data")
    label = mx.symbol.Variable(name="label")

    conv1 = mx.sym.Convolution(data, name='1/conv',
            num_filter=32, kernel=(3, 3), pad=(1, 1), no_bias=True)  # 32, 198
    concat1 = mx.sym.concat(conv1, -conv1, name='1/concat')
    bn1 = mx.sym.BatchNorm(concat1, name='1/bn',
            use_global_stats=use_global_stats, fix_gamma=False, eps=1e-04, momentum=0.99)
    pool1 = pool(bn1)

    bn2 = depthwise_conv(pool1, '2/',
            num_filter=64, kernel=(3, 3), pad=(1, 1),
            use_crelu=True, use_global_stats=use_global_stats)
    pool2 = pool(bn2)

    bn3 = depthwise_conv(pool2, '3/',
            num_filter=128, kernel=(3, 3), pad=(1, 1),
            use_global_stats=use_global_stats)
    # bn3_2 = depthwise_conv(bn3_1, '3_2/',
    #         num_filter=128, kernel=(3, 3), pad=(1, 1),
    #         use_global_stats=use_global_stats)

    groups = prepare_groups(bn3, use_global_stats)

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

    # from the original classification network
    pool6 = mx.sym.Pooling(groups[2], name='pool6', kernel=(1, 1), global_pool=True, pool_type='avg')
    fc7 = mx.sym.Convolution(pool6, name='fc7',
            num_filter=num_classes, kernel=(1, 1), pad=(0, 0), no_bias=False)
    flatten = mx.sym.flatten(fc7, name='flatten')

    softmax = mx.sym.SoftmaxOutput(flatten, name='softmax')
    return softmax
