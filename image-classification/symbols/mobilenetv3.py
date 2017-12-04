import mxnet as mx
from symbols.net_block_sep import *


def inception(data, name, f1, f3, f5, fm, do_pool, use_global_stats):
    #
    kernel = (3, 3)
    stride = (2, 2) if do_pool else (1, 1)

    # used in conv1 and convm
    pool1 = pool(data, name=name+'pool1/', kernel=(3, 3), pad=(1, 1)) if do_pool else data

    # conv1
    conv1_1 = relu_conv_bn(pool1, name+'conv1_1/',
            num_filter=f1[0], kernel=(1, 1), pad=(0, 0),
            use_global_stats=use_global_stats)
    conv1_2 = subpixel_upsample(conv1_1, f1[0]/4, 2, 2)
    conv1_3 = depthwise_conv(conv1_2, name+'conv1_3/',
            nf_dw=f1[0]/4, nf_sep=f1[1]/4,
            use_global_stats=use_global_stats)
    conv1_4 = subpixel_downsample(conv1_3, f1[1]/4, 2, 2)

    # conv3
    conv3_1 = relu_conv_bn(data, name+'conv3_1/',
            num_filter=f3[0], kernel=(1, 1), pad=(0, 0),
            use_global_stats=use_global_stats)
    conv3_2 = depthwise_conv(conv3_1, name+'conv3_2/',
            nf_dw=f3[0], nf_sep=f3[1], kernel=kernel, stride=stride,
            use_global_stats=use_global_stats)

    # conv5
    conv5_1 = relu_conv_bn(data, name+'conv5_1/',
            num_filter=f5[0], kernel=(1, 1), pad=(0, 0),
            use_global_stats=use_global_stats)
    conv5_2 = depthwise_conv(conv5_1, name+'conv5_2/',
            nf_dw=f5[0], nf_sep=f5[1], use_global_stats=use_global_stats)
    conv5_3 = depthwise_conv(conv5_2, name+'conv5_3/',
            nf_dw=f5[1], nf_sep=f5[2], kernel=kernel, stride=stride,
            use_global_stats=use_global_stats)

    # convm_1 = pool1
    convm_1 = pool(pool1, kernel=(3, 3), pad=(1, 1), name=name+'convm_1')
    convm_2 = relu_conv_bn(convm_1, name+'convm_2/',
            num_filter=fm[0], kernel=(1, 1), pad=(0, 0),
            use_global_stats=use_global_stats)
    convm_3 = depthwise_conv(convm_2, name+'convm_3/',
            nf_dw=fm[0], nf_sep=fm[1], use_global_stats=use_global_stats)
    convm_4 = subpixel_upsample(convm_3, fm[1]/4, 2, 2)

    return mx.sym.concat(conv1_4, conv3_2, conv5_3, convm_4)


def proj_add(lhs, rhs, name, num_filter, do_pool, use_global_stats):
    #
    lhs = pool(lhs, kernel=(3, 3), pad=(1, 1)) if do_pool else lhs
    lhs = relu_conv_bn(lhs, name+'lhs/',
            num_filter=num_filter, kernel=(1, 1), pad=(0, 0),
            use_global_stats=use_global_stats)
    rhs = relu_conv_bn(rhs, name+'rhs/',
            num_filter=num_filter, kernel=(1, 1), pad=(0, 0),
            use_global_stats=use_global_stats)
    return mx.sym.broadcast_add(lhs, rhs, name=name+'add/')


def topdown_feature(data, updata, name, scale, nf_proj, nf_all, nf_sqz, use_global_stats, n_mix_iter=0):
    #
    # upsample, proj, concat, mix
    updata = mx.sym.UpSampling(updata, scale=scale, sample_type='bilinear',
            num_filter=nf_all, name=name+'upsample')

    data = mx.sym.concat(data, updata, name=name+'concat')
    data = relu_conv_bn(data, name+'proj/',
            num_filter=nf_all, kernel=(1, 1), pad=(0, 0),
            use_global_stats=use_global_stats)

    data = depthwise_conv(data, name+'mix/',
            nf_dw=nf_all, nf_sep=nf_all, use_global_stats=use_global_stats)

    for i in range(n_mix_iter):
        d0 = data
        data = relu_conv_bn(data, name+'mix1x1/{}/'.format(i),
                num_filter=nf_sqz, kernel=(1, 1), pad=(0, 0),
                use_global_stats=use_global_stats)
        data = depthwise_conv(data, name+'mix3x3/{}/'.format(i),
                nf_dw=nf_sqz, nf_sep=nf_all, use_global_stats=use_global_stats)
        data = data + d0
    return data


def prepare_groups(data, use_global_stats):
    ''' prepare basic groups '''
    # 48 24 12 6 3 1
    f1 = [(32, 64), (64, 128), (128, 256)]
    f3 = [(32, 64), (64, 128), (128, 256)]
    f5 = [(32, 32, 64), (64, 64, 128), (128, 128, 256)]
    fm = [(128, 256), (256, 512), (256, 512)]
    fa = [256, 512, 1024]
    nu = [1, 3, 1]

    groups = []
    inci = data
    for i in range(3):
        inc0 = inci

        # inci = depthwise_conv(inc0, 'inc{}/init/'.format(i),
        #         nf_dw=fa[i]/2, nf_sep=fa[i], stride=(2, 2),
        #         use_global_stats=use_global_stats)
        for j in range(nu[i]):
            inci = inception(inci, 'inc{}/{}/'.format(i+3, j+1),
                    f1=f1[i], f3=f3[i], f5=f5[i], fm=fm[i], do_pool=(j == 0),
                    use_global_stats=use_global_stats)
        inci = proj_add(inc0, inci, 'inc{}/'.format(i+3),
                num_filter=fa[i], do_pool=True, use_global_stats=use_global_stats)
        groups.append(inci)

    groups[2] = depthwise_conv(groups[2], 'inc5/dil/',
            nf_dw=1024, nf_sep=512, use_global_stats=use_global_stats)

    g = groups[2]

    g = depthwise_conv(g, 'g3/1/',
            nf_dw=512, nf_sep=512, kernel=(3, 3), pad=(1, 1), stride=(2, 2),
            use_global_stats=use_global_stats)
    g = depthwise_conv(g, 'g3/2/',
            nf_dw=512, nf_sep=512, use_global_stats=use_global_stats)
    groups.append(g)

    g = depthwise_conv(g, 'g4/1/',
            nf_dw=512, nf_sep=512, kernel=(3, 3), pad=(1, 1), stride=(2, 2),
            use_global_stats=use_global_stats)
    g = depthwise_conv(g, 'g4/2/',
            nf_dw=512, nf_sep=512, use_global_stats=use_global_stats)
    groups.append(g)

    g = depthwise_conv(g, 'g5/1/',
            nf_dw=512, nf_sep=512, pad=(0, 0), use_global_stats=use_global_stats)
    groups.append(g)

    # top-down features
    groups[0] = topdown_feature(groups[0], groups[1], 'up0/', scale=2,
            nf_proj=128, nf_all=512, nf_sqz=256, use_global_stats=use_global_stats)

    return groups


def get_symbol(num_classes=1000, **kwargs):
    '''
    '''
    use_global_stats = kwargs['use_global_stats']

    data = mx.symbol.Variable(name="data")
    label = mx.symbol.Variable(name="label")

    conv1 = mx.sym.Convolution(data, name='1/conv',
            num_filter=16, kernel=(4, 4), pad=(1, 1), stride=(2, 2), no_bias=True)  # 32, 198
    concat1 = mx.sym.concat(conv1, -conv1, name='1/concat')
    bn1 = mx.sym.BatchNorm(concat1, name='1/bn', use_global_stats=use_global_stats, fix_gamma=False)

    bn2 = depthwise_conv(bn1, '2_1/',
            nf_dw=32, nf_sep=64, use_global_stats=use_global_stats)
    bn2 = depthwise_conv(bn2, '2_2/',
            nf_dw=64, nf_sep=64, kernel=(4, 4), pad=(1, 1), stride=(2, 2),
            use_global_stats=use_global_stats)

    bn3 = depthwise_conv(bn2, '3_1/',
            nf_dw=64, nf_sep=128, use_global_stats=use_global_stats)
    bn3 = depthwise_conv(bn3, '3_2/',
            nf_dw=128, nf_sep=128, use_global_stats=use_global_stats)

    groups = prepare_groups(bn3, use_global_stats)

    # hyper_groups = []
    # nf_hyper = [(320, 192) for _ in groups]
    #
    # for i, (g, nf) in enumerate(zip(groups, nf_hyper)):
    #     p1 = relu_conv_bn(g, 'hypercls{}/1/'.format(i),
    #             num_filter=nf[0], kernel=(1, 1), pad=(0, 0),
    #             use_global_stats=use_global_stats)
    #     # p1 = depthwise_conv(p1, 'hypercls{}/2/'.format(i),
    #     #         nf_dw=nf[0], nf_sep=nf[0],
    #     #         use_global_stats=use_global_stats)
    #     # p1 = relu_conv_bn(p1, 'hypercls{}/2/'.format(i),
    #     #         num_filter=nf[0], kernel=(1, 1), pad=(0, 0),
    #     #         use_global_stats=use_global_stats)
    #     h1 = mx.sym.Activation(p1, name='hyper{}/1'.format(i), act_type='relu')
    #
    #     p2 = relu_conv_bn(g, 'hyperreg{}/1/'.format(i),
    #             num_filter=nf[1], kernel=(1, 1), pad=(0, 0),
    #             use_global_stats=use_global_stats)
    #     # p2 = depthwise_conv(p2, 'hyperreg{}/2/'.format(i),
    #     #         nf_dw=nf[1], nf_sep=nf[1],
    #     #         use_global_stats=use_global_stats)
    #     # p2 = relu_conv_bn(p2, 'hyperreg{}/2/'.format(i),
    #     #         num_filter=nf[1], kernel=(1, 1), pad=(0, 0),
    #     #         use_global_stats=use_global_stats)
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
    flatten = mx.sym.Flatten(fc7, name='flatten')

    softmax = mx.sym.SoftmaxOutput(flatten, name='softmax')
    return softmax
