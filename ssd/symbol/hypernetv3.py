import mxnet as mx
from symbol.net_block import *


def prepare_groups(group_i, n_curr_ch, use_global_stats):
    ''' prepare basic groups '''
    # 48 24 12 6 3
    nf_3x3 = [(80, 40), (96, 48), (128, 64), (96, 48), (96,)]
    nf_1x1 = [40, 48, 64, 48, 48]
    n_unit = [2, 3, 2, 1, 1]

    nf_dn = [32, 32, 32, 64, 64]

    # prepare groups
    n_group = len(nf_1x1)
    groups = []
    dn_groups = [[] for _ in nf_1x1]
    for i, (nf3, nf1) in enumerate(zip(nf_3x3, nf_1x1)):
        if nf_dn[i] > 0:
            d = relu_conv_bn(group_i, 'dn{}{}/'.format(i-1, i),
                    num_filter=nf_dn[i], kernel=(3, 3), pad=(1, 1), stride=(2, 2),
                    use_global_stats=use_global_stats)
            dn_groups[i].append(d)

        group_i = pool(group_i)
        group_i = relu_conv_bn(group_i, 'gp{}/'.format(i), num_filter=sum(nf3)+nf1,
                kernel=(1, 1), pad=(0, 0),
                use_global_stats=use_global_stats)
        for j in range(n_unit[i]):
            prefix_name = 'g{}/u{}/'.format(i, j)
            g = conv_group(group_i, prefix_name,
                    num_filter_3x3=nf3, num_filter_1x1=nf1, do_proj=True,
                    use_global_stats=use_global_stats)
            group_i = g + group_i
        # group_i = proj_add(gp, group_i, nfa, use_global_stats)
        groups.append(group_i)
    # dn_groups[-1].append( \
    #         relu_conv_bn(groups[2], 'dn24/',
    #             num_filter=nf_dn[-1], kernel=(5, 5), pad=(2, 2), stride=(4, 4),
    #             use_global_stats=use_global_stats))

    up_groups = [[] for _ in groups]
    for i, g in enumerate(groups[1:3]):
        up_groups[i].append( \
                upsample_feature(g, name='up{}{}/'.format(i+1, i), scale=2,
                    num_filter_proj=32, num_filter_upsample=32, use_global_stats=use_global_stats))
    up_groups[0].append( \
            upsample_feature(groups[2], name='up20/', scale=4,
                num_filter_proj=64, num_filter_upsample=32, use_global_stats=use_global_stats))

    return groups, dn_groups, up_groups


def get_symbol(num_classes=1000, **kwargs):
    '''
    '''
    use_global_stats = kwargs['use_global_stats']

    data = mx.symbol.Variable(name="data")
    label = mx.symbol.Variable(name="label")

    conv1 = convolution(data, name='1/conv',
        num_filter=32, kernel=(4, 4), pad=(1, 1), stride=(2, 2), no_bias=True)  # 32, 198
    concat1 = mx.sym.concat(conv1, -conv1, name='1/concat')
    bn1 = batchnorm(concat1, name='1/bn', use_global_stats=use_global_stats, fix_gamma=False)
    pool1 = pool(bn1)

    bn2 = relu_conv_bn(pool1, '2/',
            num_filter=32, kernel=(3, 3), pad=(1, 1), stride=(2, 2), use_crelu=True,
            use_global_stats=use_global_stats)

    bn3_1 = conv_group(bn2, '3/1/',
            num_filter_3x3=(64, 32), num_filter_1x1=32, do_proj=True,
            use_global_stats=use_global_stats)
    bn3_2 = conv_group(bn3_1, '3/2/',
            num_filter_3x3=(64, 32), num_filter_1x1=32, do_proj=True,
            use_global_stats=use_global_stats)
    bn3 = bn3_1 + bn3_2

    n_curr_ch = 128
    groups, dn_groups, up_groups = prepare_groups(bn3, n_curr_ch, use_global_stats)

    hyper_groups = []
    nf_hyper = [256, 256, 256, 256, 192]

    for i, (g, u, d) in enumerate(zip(groups, up_groups, dn_groups)):
        p = mx.sym.concat(*([g] + d + u))
        p1 = relu_conv_bn(p, 'hyperc1{}/'.format(i),
                num_filter=nf_hyper[i], kernel=(1, 1), pad=(0, 0),
                use_global_stats=use_global_stats)
        p2 = relu_conv_bn(p, 'hyperc2{}/'.format(i),
                num_filter=nf_hyper[i], kernel=(1, 1), pad=(0, 0),
                use_global_stats=use_global_stats)
        h1 = mx.sym.Activation(p1, name='hyper{}/1'.format(i), act_type='relu')
        h2 = mx.sym.Activation(p2, name='hyper{}/2'.format(i), act_type='relu')
        hyper_groups.append((h1, h2))

    pooled = []
    for i, h in enumerate(hyper_groups):
        hc = mx.sym.concat(h[0], h[1])
        p = mx.sym.Pooling(hc, kernel=(2, 2), global_pool=True, pool_type='max')
        pooled.append(p)

    pooled_all = mx.sym.flatten(mx.sym.concat(*pooled), name='flatten')
    fc1 = mx.sym.FullyConnected(pooled_all, num_hidden=4096, name='fc1')
    softmax = mx.sym.SoftmaxOutput(data=fc1, label=label, name='softmax')
    return softmax
