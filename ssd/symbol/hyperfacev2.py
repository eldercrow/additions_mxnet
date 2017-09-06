import mxnet as mx
from symbol.net_block import *


def prepare_groups(group_i, use_global_stats):
    ''' prepare basic groups '''
    # 96 48 24 12 6 3
    nf_3x3 = [(24, 24), (48, 48), (64, 64), (64, 64), (32, 32), (64,)]
    nf_1x1 = [48, 96, 128, 128, 64, 64]
    n_unit = [2, 2, 2, 2, 1, 1]

    # prepare groups
    n_group = len(nf_1x1)
    groups = []
    ctx_groups = []
    for i, (nf3, nf1) in enumerate(zip(nf_3x3, nf_1x1)):
        g0 = pool(group_i)
        g = g0
        for j in range(n_unit[i]):
            prefix_name = 'g{}/u{}/'.format(i, j)
            g = conv_group(g, prefix_name,
                    num_filter_3x3=nf3, num_filter_1x1=nf1, do_proj=False,
                    use_global_stats=use_global_stats)
        group_i = proj_add(g, g0, 'g{}/'.format(i), sum(nf3)+nf1, use_global_stats)
        groups.append(group_i)

    up_groups = [[] for _ in groups]
    for i in range(3):
        ss = 2
        for j in range(i+1, 4):
            nfu = 32
            u = upsample_feature(groups[j], name='up{}{}/'.format(j, i), scale=ss,
                    num_filter_proj=32, num_filter_upsample=nfu, use_global_stats=use_global_stats)
            up_groups[i].append(u)
            ss *= 2

    return groups, up_groups


def get_symbol(num_classes=1000, **kwargs):
    '''
    '''
    use_global_stats = kwargs['use_global_stats']

    data = mx.symbol.Variable(name="data")
    label = mx.symbol.Variable(name="label")

    conv1 = convolution(data, name='1/conv',
        num_filter=12, kernel=(3, 3), pad=(1, 1), no_bias=True)  # 32, 198
    concat1 = mx.sym.concat(conv1, -conv1, name='1/concat')
    bn1 = batchnorm(concat1, name='1/bn', use_global_stats=use_global_stats, fix_gamma=False)
    pool1 = pool(bn1)

    bn2 = relu_conv_bn(pool1, '2/',
            num_filter=24, kernel=(3, 3), pad=(1, 1), use_crelu=True,
            use_global_stats=use_global_stats)
    # pool2 = pool(bn2)

    # bn3 = conv_group(pool2, '3/',
    #         num_filter_3x3=(24, 24), num_filter_1x1=48, do_proj=False,
    #         use_global_stats=use_global_stats)
    # bn3 = proj_add(pool2, bn3, '3/', 96, use_global_stats)

    groups, up_groups = prepare_groups(bn2, use_global_stats)

    nf_proj = [32, 64, 96, 128, 96, 96]
    nf_concat = [256, 256, 256, 256, 192, 192]
    hyper_groups = []
    for i, (g, u, nf) in enumerate(zip(groups, up_groups, nf_proj)):
        g = relu_conv_bn(g, prefix_name='gp{}/'.format(i),
                num_filter=nf, kernel=(1, 1), pad=(0, 0),
                use_global_stats=use_global_stats)
        c = mx.sym.concat(*([g] + u))
        c = relu_conv_bn(c, prefix_name='hc{}/'.format(i),
                num_filter=nf_concat[i], kernel=(3, 3), pad=(1, 1),
                use_global_stats=use_global_stats)
        groups[i] = c

    nf_hyper = [128] * 6
    for i, g in enumerate(groups):
        p1 = relu_conv_bn(g, 'hyperc1{}/'.format(i),
                num_filter=nf_hyper[i], kernel=(1, 1), pad=(0, 0),
                use_global_stats=use_global_stats)
        p2 = relu_conv_bn(g, 'hyperc2{}/'.format(i),
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
    # softmax = mx.sym.SoftmaxOutput(data=pooled_all, label=label, name='softmax')
    fc1 = mx.sym.FullyConnected(pooled_all, num_hidden=4096, name='fc1')
    softmax = mx.sym.SoftmaxOutput(data=fc1, label=label, name='softmax')
    return softmax
