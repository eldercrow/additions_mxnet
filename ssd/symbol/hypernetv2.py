import mxnet as mx
from symbol.net_block import *


def prepare_groups(group_i, n_curr_ch, use_global_stats):
    ''' prepare basic groups '''
    # feat_sz = [56, 28, 14, 7, 3, 1]
    feat_sz = [48, 24, 12, 6, 3, 1]
    nf_3x3 = [(32, 32), (48, 48), (64, 64), (64, 32), (96,), ()]
    nf_1x1 = [128, 192, 192, 192, 192, 192]
    nf_init = [64, 96, 128, 96, 96, 192]
    nf_down = [64, 64, 64, 64, 64, 128]
    use_init = [True, True, True, True, True, True]
    n_units = [3, 4, 3, 2, 1, 1]

    groups = []
    dn_groups = []
    up_groups = [[] for _ in nf_init[:-1]]

    for i, (nf3, nf1, nfi, nfd) in enumerate(zip(nf_3x3, nf_1x1, nf_init, nf_down)):
        if i < 5:
            d = relu_conv_bn(group_i, 'dn{}/'.format(i),
                    num_filter=nfd, kernel=(3, 3), pad=(1, 1), stride=(2, 2),
                    use_global_stats=use_global_stats)
            group_i = pool(group_i)
        else:
            d = relu_conv_bn(group_i, 'dn{}/'.format(i),
                    num_filter=nfd, kernel=(3, 3), pad=(0, 0), stride=(2, 2),
                    use_global_stats=use_global_stats)
            group_i = pool(group_i, kernel=(3, 3))
        dn_groups.append(d)

        for j in range(n_units[i]):
            group_i, n_curr_ch = inception_group(group_i, 'g{}u{}/'.format(i, j), n_curr_ch,
                    num_filter_3x3=nf3, num_filter_1x1=nf1, num_filter_init=nfi,
                    use_init=use_init[i], use_global_stats=use_global_stats)

        groups.append(group_i)

        if i > 0:
            s = feat_sz[i-1] / feat_sz[i]
            u = upsample_feature(group_i, 'up{}{}/'.format(i, i-1), scale=s,
                    num_filter_proj=64, num_filter_upsample=64,
                    use_global_stats=use_global_stats)
            up_groups[i-1].append(u)

    up_groups[0].append(\
            upsample_feature(groups[2], 'up20/', scale=4,
                    num_filter_proj=128, num_filter_upsample=64,
                    use_global_stats=use_global_stats))
    up_groups.append([])

    return groups, dn_groups, up_groups


def get_symbol(num_classes=1000, **kwargs):
    '''
    '''
    use_global_stats = kwargs['use_global_stats']

    data = mx.symbol.Variable(name="data")
    label = mx.symbol.Variable(name="label")

    conv1 = convolution(data, name='1/conv',
        num_filter=24, kernel=(4, 4), pad=(1, 1), stride=(2, 2), no_bias=True)  # 32, 198
    concat1 = mx.sym.concat(conv1, -conv1, name='1/concat')
    bn1 = batchnorm(concat1, name='1/bn', use_global_stats=use_global_stats, fix_gamma=False)
    pool1 = bn1

    bn2_1 = relu_conv_bn(pool1, '2_1/', num_filter=24, kernel=(3, 3), pad=(1, 1),
            use_crelu=True, use_global_stats=use_global_stats)
    bn2_2 = relu_conv_bn(pool1, '2_2/', num_filter=24, kernel=(3, 3), pad=(1, 1),
            use_global_stats=use_global_stats)
    bn2 = mx.sym.concat(bn2_1, bn2_2, name='2/concat')
    pool2 = pool(bn2)
    n_curr_ch = 72

    bn3_1 = pool2
    for i in range(2):
        bn3_1, n_curr_ch = inception_group(bn3_1, '3/{}/'.format(i), n_curr_ch,
                num_filter_3x3=(24, 24), num_filter_1x1=96, num_filter_init=48, use_init=True,
                use_global_stats=use_global_stats)
    bn3_2 = relu_conv_bn(bn2, '3d/', num_filter=32, kernel=(3, 3), pad=(1, 1), stride=(2, 2),
            use_global_stats=use_global_stats)
    bn3 = mx.sym.concat(bn3_1, bn3_2, name='3/concat')
    n_curr_ch = 128

    groups, dn_groups, up_groups = prepare_groups(bn3, n_curr_ch, use_global_stats)

    hgroups = []
    nf_hyper = [320, 320, 320, 320, 256, 256]
    for i, (g, d, u) in enumerate(zip(groups, dn_groups, up_groups)):
        h = [g, d] + u
        c = mx.sym.concat(*h)

        hc1 = relu_conv_bn(c, 'hyperc1/{}/'.format(i),
                num_filter=nf_hyper[i], kernel=(1, 1), pad=(0, 0),
                use_global_stats=use_global_stats)
        hc2 = relu_conv_bn(c, 'hyperc2/{}/'.format(i),
                num_filter=nf_hyper[i], kernel=(1, 1), pad=(0, 0),
                use_global_stats=use_global_stats)
        hyper1 = mx.sym.Activation(hc1, name='hyper{}/1'.format(i), act_type='relu')
        hyper2 = mx.sym.Activation(hc2, name='hyper{}/2'.format(i), act_type='relu')
        hgroups.append((hyper1, hyper2))

    pooled = []
    for i, h in enumerate(hgroups):
        hc = mx.sym.concat(h[0], h[1])
        p = mx.sym.Pooling(hc, kernel=(2,2), global_pool=True, pool_type='max')
        pooled.append(p)

    pooled_all = mx.sym.flatten(mx.sym.concat(*pooled), name='flatten')
    # softmax = mx.sym.SoftmaxOutput(data=pooled_all, label=label, name='softmax')
    fc1 = mx.sym.FullyConnected(pooled_all, num_hidden=4096, name='fc1')
    softmax = mx.sym.SoftmaxOutput(data=fc1, label=label, name='softmax')
    return softmax
