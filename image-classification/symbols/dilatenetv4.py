import mxnet as mx
from symbols.net_block import *


def prepare_groups(data, use_global_stats):
    ''' prepare basic groups '''
    # 96 48 24 12 6 3
    dilates = [2, 3, 4, 5, 6, 7]
    nf_dil = [64, 64, 40, 40, 24, 24]
    groups = []
    group_i = data
    for i, (nf, dil) in enumerate(zip(nf_dil, dilates)):
        dilate = (dil, dil)
        pad = dilate
        group_i = relu_conv_bn(group_i, 'gd{}/'.format(i),
                num_filter=nf, kernel=(3, 3), pad=pad, dilate=dilate,
                use_global_stats=use_global_stats)
        groups.append(group_i)

    nf_all = sum(nf_dil)
    nf_sqz = nf_all / 2
    group_all = mx.sym.concat(*groups)
    g = relu_conv_bn(group_all, 'gall/',
            num_filter=nf_all, kernel=(1, 1), pad=(0, 0),
            use_global_stats=use_global_stats)
    g = g + data

    for j in range(1):
        gp = g
        g = relu_conv_bn(g, 'g0/{}/1x1/'.format(j),
                num_filter=nf_sqz, kernel=(1, 1), pad=(0, 0),
                use_global_stats=use_global_stats)
        g = relu_conv_bn(g, 'g0/{}/3x3/'.format(j),
                num_filter=nf_all, kernel=(3, 3), pad=(1, 1),
                use_global_stats=use_global_stats)
        g = g + gp

    groups = [g]
    n_units = (2, 2, 2, 1)
    for i, nu in enumerate(n_units, 1):
        if i == 3:
            # following the original ssd
            g = relu_conv_bn(g, 'g{}/'.format(i),
                    num_filter=nf_all, kernel=(3, 3), pad=(2, 2), dilate=(2, 2),
                    use_global_stats=use_global_stats)

        g = pool(g)
        for j in range(nu):
            gp = g
            g = relu_conv_bn(g, 'g{}/{}/1x1/'.format(i, j),
                    num_filter=nf_sqz, kernel=(1, 1), pad=(0, 0),
                    use_global_stats=use_global_stats)
            g = relu_conv_bn(g, 'g{}/{}/3x3/'.format(i, j),
                    num_filter=nf_all, kernel=(3, 3), pad=(1, 1),
                    use_global_stats=use_global_stats)
            g = g + gp
        groups.append(g)

    g = relu_conv_bn(g, 'g5/1x1/'.format(i),
            num_filter=nf_sqz, kernel=(1, 1), pad=(0, 0),
            use_global_stats=use_global_stats)
    g = relu_conv_bn(g, 'g5/3x3/'.format(i),
            num_filter=nf_all, kernel=(3, 3), pad=(0, 0),
            use_global_stats=use_global_stats)
    # g = g + gp
    groups.append(g)

    return groups


# def mix_groups(groups, use_global_stats):
#     ''' divide each group and mix '''
#     nf_block = 96
#     n_group = len(groups) # 6, in general
#
#     # downsample features
#     dn_groups = [[] for _ in groups]
#     for i, g in enumerate(groups[:-1], 1):
#         kernel, pad = ((3, 3), (0, 0)) if i == n_group-1 else ((4, 4), (1, 1))
#
#         d = relu_conv_bn(g, 'dp{}/'.format(i),
#                 num_filter=nf_block, kernel=(1, 1), pad=(0, 0),
#                 use_global_stats=use_global_stats)
#         d = relu_conv_bn(d, 'dn{}/'.format(i),
#                 num_filter=nf_block, kernel=kernel, pad=pad, stride=(2, 2),
#                 use_global_stats=use_global_stats)
#         dn_groups[i].append(d)
#
#     # upsample features
#     up_groups = [[] for _ in groups]
#     for i, g in enumerate(groups[1:]):
#         scale = 3 if i == n_group-2 else 2
#
#         u = upsample_feature(g, 'up{}/'.format(i), scale=scale,
#                 num_filter_proj=nf_block, num_filter_upsample=nf_block,
#                 use_global_stats=use_global_stats)
#         up_groups[i].append(u)
#
#     nf_main = [nf_block for _ in groups]
#     nf_main[0] *= 2
#     nf_main[-1] *= 2
#     for i, (g, u, d, nf) in enumerate(zip(groups, up_groups, dn_groups, nf_main)):
#         g = relu_conv_bn(g, 'ct1x1/{}/'.format(i),
#                 num_filter=nf, kernel=(1, 1), pad=(0, 0),
#                 use_global_stats=use_global_stats)
#         g = relu_conv_bn(g, 'ct3x3/{}/'.format(i),
#                 num_filter=nf, kernel=(3, 3), pad=(1, 1),
#                 use_global_stats=use_global_stats)
#         groups[i] = mx.sym.concat(*([g] + u + d))
#
#     return groups


def get_symbol(num_classes=1000, **kwargs):
    '''
    '''
    use_global_stats = kwargs['use_global_stats']

    data = mx.symbol.Variable(name="data")
    label = mx.symbol.Variable(name="softmax_label")

    # data = mx.sym.Custom(data, name='data_dummy', op_type='dummy')

    conv1 = convolution(data, name='1/conv',
        num_filter=24, kernel=(4, 4), pad=(1, 1), stride=(2, 2), no_bias=True)  # 32, 198
    concat1 = mx.sym.concat(conv1, -conv1, name='1/concat')
    bn1 = batchnorm(concat1, name='1/bn', use_global_stats=use_global_stats, fix_gamma=False)
    pool1 = bn1 #pool(bn1)

    bn2_1 = relu_conv_bn(pool1, '2_1/',
            num_filter=24, kernel=(3, 3), pad=(1, 1), use_crelu=True,
            use_global_stats=use_global_stats)
    bn2_2 = relu_conv_bn(pool1, '2_2/',
            num_filter=16, kernel=(3, 3), pad=(1, 1),
            use_global_stats=use_global_stats)
    bn2 = mx.sym.concat(bn2_1, bn2_2)
    pool2 = pool(bn2)

    bn3_1 = relu_conv_bn(pool2, '3_1/',
            num_filter=24, kernel=(3, 3), pad=(1, 1), use_crelu=True,
            use_global_stats=use_global_stats)
    bn3_2 = relu_conv_bn(pool2, '3_2/',
            num_filter=80, kernel=(3, 3), pad=(1, 1),
            use_global_stats=use_global_stats)
    bn3 = mx.sym.concat(bn3_1, bn3_2)
    pool3 = pool(bn3)

    bn4_1 = relu_conv_bn(pool3, '4_1/',
            num_filter=128, kernel=(3, 3), pad=(1, 1),
            use_global_stats=use_global_stats)
    bn4_2 = relu_conv_bn(bn4_1, '4_2/',
            num_filter=128, kernel=(3, 3), pad=(1, 1),
            use_global_stats=use_global_stats)
    bn4 = mx.sym.concat(bn4_1, bn4_2)
    bn4 = relu_conv_bn(bn4, '4/',
            num_filter=256, kernel=(1, 1), pad=(0, 0),
            use_global_stats=use_global_stats)

    groups = prepare_groups(bn4, use_global_stats)

    hyper_groups = []
    nf_hyper = [192 for _ in groups] #[192, 192, 192, 192, 192, 192]

    for i, (g, nf) in enumerate(zip(groups, nf_hyper)):
        p1 = relu_conv_bn(g, 'hyperc1/1/{}/'.format(i),
                num_filter=nf, kernel=(1, 1), pad=(0, 0),
                use_global_stats=use_global_stats)
        h1 = mx.sym.Activation(p1, name='hyper{}/1'.format(i), act_type='relu')

        p2 = relu_conv_bn(g, 'hyperc2/1/{}/'.format(i),
                num_filter=nf, kernel=(1, 1), pad=(0, 0),
                use_global_stats=use_global_stats)
        h2 = mx.sym.Activation(p2, name='hyper{}/2'.format(i), act_type='relu')
        hyper_groups.append((h1, h2))
        # hyper_groups.append(h1)

    pooled = []
    ps = 8
    for i, h in enumerate(hyper_groups[:-2]):
        h = mx.sym.concat(h[0], h[1])
        if ps > 1:
            p = mx.sym.Pooling(h, kernel=(ps, ps), stride=(ps, ps), pool_type='max')
        else:
            p = h
        ps /= 2
        pooled.append(p)

    pooled_all = mx.sym.flatten(mx.sym.concat(*pooled), name='flatten')
    fc1 = mx.sym.FullyConnected(pooled_all, num_hidden=4096, name='fc1', no_bias=True)
    bn_fc1 = mx.sym.BatchNorm(fc1, use_global_stats=use_global_stats, fix_gamma=False)
    relu_fc1 = mx.sym.Activation(bn_fc1, act_type='relu')
    fc2 = mx.sym.FullyConnected(relu_fc1, num_hidden=num_classes, name='fc2')
    cls_prob = mx.sym.softmax(fc2, name='cls_prob')
    softmax = mx.sym.SoftmaxOutput(data=fc2, label=label, name='softmax')
    return softmax
