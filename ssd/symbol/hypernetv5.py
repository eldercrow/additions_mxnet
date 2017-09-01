import mxnet as mx
from symbol.net_block_preact import *


def prepare_groups(group_i, n_curr_ch, use_global_stats):
    ''' prepare basic groups '''
    # 48 24 12 6 3
    nf_3x3 = [(128, 64), (256, 128), (384, 192), (128, 64), (128,), ()]
    nf_1x1 = [64, 128, 192, 64, 128, 192]
    n_unit = [2, 2, 2, 1, 1, 1]

    nf_dn = [0, 0, 0, 128, 128, 192]

    # prepare groups
    n_group = len(nf_1x1)
    groups = []
    dn_groups = [[] for _ in nf_1x1]
    for i, (nf3, nf1) in enumerate(zip(nf_3x3, nf_1x1)):
        if nf_dn[i] > 0:
            pad = (1, 1) if i < 5 else (0, 0)
            d = bn_relu_conv(group_i, 'dn{}{}/'.format(i-1, i),
                    num_filter=nf_dn[i], kernel=(3, 3), pad=pad, stride=(2, 2),
                    use_global_stats=use_global_stats)
            dn_groups[i].append(d)

        kernel = (2, 2) if i < 5 else (3, 3)
        group_i = pool(group_i, kernel=kernel)
        g0 = group_i
        for j in range(n_unit[i]):
            prefix_name = 'g{}/u{}/'.format(i, j)
            group_i = conv_group(group_i, prefix_name,
                    num_filter_3x3=nf3, num_filter_1x1=nf1, do_proj=False,
                    use_global_stats=use_global_stats)
        group_i = proj_add(g0, group_i, 'g{}/'.format(i),
                num_filter=sum(nf3)+nf1, use_global_stats=use_global_stats)
        groups.append(group_i)

    up_groups = [[] for _ in groups]
    for i, g in enumerate(groups[1:3]):
        up_groups[i].append( \
                upsample_feature(g, name='up{}{}/'.format(i+1, i), scale=2,
                    num_filter_proj=128, num_filter_upsample=0, use_global_stats=use_global_stats))
    up_groups[0].append( \
            upsample_feature(groups[2], name='up20/', scale=4,
                num_filter_proj=128, num_filter_upsample=0, use_global_stats=use_global_stats))

    return groups, dn_groups, up_groups


def get_symbol(num_classes=1000, **kwargs):
    '''
    '''
    use_global_stats = kwargs['use_global_stats']

    data = mx.symbol.Variable(name="data")
    label = mx.symbol.Variable(name="label")

    conv1 = mx.sym.Convolution(data, name='1/conv',
        num_filter=16, kernel=(3, 3), pad=(1, 1), no_bias=True)  # 32, 198
    concat1 = mx.sym.concat(conv1, -conv1, name='1/concat')
    pool1 = pool(concat1)

    bn2_1 = bn_relu_conv(pool1, '2_1/',
            num_filter=24, kernel=(3, 3), pad=(1, 1), use_crelu=True,
            use_global_stats=use_global_stats)
    bn2_2 = bn_relu_conv(pool1, '2_2/',
            num_filter=16, kernel=(3, 3), pad=(1, 1),
            use_global_stats=use_global_stats)
    bn2 = mx.sym.concat(bn2_1, bn2_2)
    pool2 = pool(bn2)

    bn3_1 = conv_group(pool2, '3/1/',
            num_filter_3x3=(64, 32), num_filter_1x1=32, do_proj=False,
            use_global_stats=use_global_stats)
    bn3 = proj_add(pool2, bn3_1, '3/', num_filter=128, use_global_stats=use_global_stats)

    n_curr_ch = 128
    groups, dn_groups, up_groups = prepare_groups(bn3, n_curr_ch, use_global_stats)

    hyper_groups = []
    nf_hyper = [256, 256, 256, 256, 256, 192]
    # nf_hyper_proj = [64] * 5

    for i, (g, u, d) in enumerate(zip(groups, up_groups, dn_groups)):
        p = mx.sym.concat(*([g] + d + u))
        # p = bn_relu_conv(p, 'hyperp{}/'.format(i),
        #         num_filter=nf_hyper_proj[i], kernel=(1, 1), pad=(0, 0),
        #         use_global_stats=use_global_stats)
        h1 = bn_relu_conv(p, 'hyperc{}/1/'.format(i),
                num_filter=nf_hyper[i], kernel=(1, 1), pad=(0, 0),
                use_global_stats=use_global_stats)
        h1 = mx.sym.BatchNorm(h1, use_global_stats=use_global_stats, fix_gamma=False)
        h1 = mx.sym.Activation(h1, name='hyper{}/1'.format(i), act_type='relu')

        h2 = bn_relu_conv(p, 'hyperc{}/2/'.format(i),
                num_filter=nf_hyper[i], kernel=(1, 1), pad=(0, 0),
                use_global_stats=use_global_stats)
        h2 = mx.sym.BatchNorm(h2, use_global_stats=use_global_stats, fix_gamma=False)
        h2 = mx.sym.Activation(h2, name='hyper{}/2'.format(i), act_type='relu')
        hyper_groups.append((h1, h2))

    pooled = []
    for i, h in enumerate(hyper_groups):
        hc = mx.sym.concat(h[0], h[1])
        p = mx.sym.Pooling(hc, kernel=(2, 2), global_pool=True, pool_type='max')
        pooled.append(p)

    pooled_all = mx.sym.flatten(mx.sym.concat(*pooled), name='flatten')
    # softmax = mx.sym.SoftmaxOutput(data=pooled_all, label=label, name='softmax')
    fc1 = mx.sym.FullyConnected(pooled_all, num_hidden=4096, name='fc1')
    softmax = mx.sym.SoftmaxOutput(data=fc1, label=label, name='softmax')
    return softmax
