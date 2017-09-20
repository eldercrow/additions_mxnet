import mxnet as mx
from symbol.net_block import *


def prepare_groups(group_i, use_global_stats):
    ''' prepare basic groups '''
    # 96 48 24 12 6 3
    nf_3x3 = [(32, 16), (48, 24), (64, 32), (64, 32), (48, 24), (48,)]
    nf_1x1 = [16, 24, 32, 32, 24, 48]
    n_unit = [2, 2, 3, 3, 1, 1]

    # prepare groups
    n_group = len(nf_1x1)
    groups = []
    for i, (nf3, nf1) in enumerate(zip(nf_3x3, nf_1x1)):
        group_i = pool(group_i)
        g0 = group_i
        for j in range(n_unit[i]):
            prefix_name = 'g{}/u{}/'.format(i, j)
            group_i = conv_group(group_i, prefix_name,
                    num_filter_3x3=nf3, num_filter_1x1=nf1,
                    use_global_stats=use_global_stats)
        group_i = proj_add(g0, group_i, 'g{}/res/'.format(i), sum(nf3)+nf1, use_global_stats)
        groups.append([group_i])

    # upsample for g0
    ss = 2
    for i, g in enumerate(groups[1:4], 1):
        nf_proj = 32 * ss * ss / 4
        nf_up = 32 / ss * 2
        if i == 4:
            nf_proj, nf_up = nf_proj / 2, nf_up / 2
        u = upsample_feature(g[0], name='up{}0/'.format(i), scale=ss,
                num_filter_proj=nf_proj, num_filter_upsample=nf_up, use_global_stats=use_global_stats)
        groups[0].append(u)
        ss *= 2
    # upsample for g1
    ss = 2
    for i, g in enumerate(groups[2:4], 1):
        nf_proj = 32 * ss * ss / 4
        nf_up = 32 / ss * 2
        u = upsample_feature(g[0], name='up{}1/'.format(i), scale=ss,
                num_filter_proj=nf_proj, num_filter_upsample=nf_up, use_global_stats=use_global_stats)
        groups[1].append(u)
        ss *= 2

    return groups


def get_symbol(num_classes=1000, **kwargs):
    '''
    '''
    use_global_stats = kwargs['use_global_stats']

    data = mx.symbol.Variable(name="data")
    label = mx.symbol.Variable(name="label")

    conv1 = convolution(data, name='1/conv',
        num_filter=16, kernel=(3, 3), pad=(1, 1), no_bias=True)  # 32, 198
    concat1 = mx.sym.concat(conv1, -conv1, name='1/concat')
    bn1 = batchnorm(concat1, name='1/bn', use_global_stats=use_global_stats, fix_gamma=False)
    pool1 = pool(bn1)

    bn2 = relu_conv_bn(pool1, '2/', num_filter=24,
            kernel=(3, 3), pad=(1, 1), use_global_stats=use_global_stats)

    groups = prepare_groups(bn2, use_global_stats)

    nf_group = [192, 192, 192, 192, 144, 144]
    for i, (g, nf) in enumerate(zip(groups, nf_group)):
        g = mx.sym.concat(*g, name='gc{}/'.format(i)) if len(g) > 1 else g[0]
        g = relu_conv_bn(g, 'gc1x1{}/'.format(i),
                num_filter=nf/2, kernel=(1, 1), pad=(0, 0),
                use_global_stats=use_global_stats)
        g = relu_conv_bn(g, 'gc3x3{}/'.format(i),
                num_filter=nf, kernel=(3, 3), pad=(1, 1),
                use_global_stats=use_global_stats)
        groups[i] = g

    hyper_groups = []
    nf_hyper = [96, 96, 96, 96, 72, 72]

    for i, (g, nf) in enumerate(zip(groups, nf_hyper)):
        p1 = relu_conv_bn(g, 'hyperc1{}/'.format(i),
                num_filter=nf, kernel=(1, 1), pad=(0, 0),
                use_global_stats=use_global_stats)
        p2 = relu_conv_bn(g, 'hyperc2{}/'.format(i),
                num_filter=nf, kernel=(1, 1), pad=(0, 0),
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
    # softmax = mx.sym.SoftmaxOutput(data=pooled_all, label=label, name='softmax')
    fc1 = mx.sym.FullyConnected(pooled_all, num_hidden=4096, name='fc1')
    softmax = mx.sym.SoftmaxOutput(data=fc1, label=label, name='softmax')
    return softmax
