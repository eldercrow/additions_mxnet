import mxnet as mx
from symbol.net_block import *


def prepare_groups(group_i, n_curr_ch, use_global_stats):
    ''' prepare basic groups '''
    # 96 48 24 12 6 3
    nf_3x3 = [(48, 24), (64, 32), (80, 40), (80, 40), (48, 24), (48,)]
    nf_1x1 = [24, 32, 40, 40, 24, 48]
    n_unit = [2, 2, 2, 2, 1, 1]

    # prepare groups
    n_group = len(nf_1x1)
    groups = []
    for i, (nf3, nf1) in enumerate(zip(nf_3x3, nf_1x1)):
        group_i = pool(group_i)
        for j in range(n_unit[i]):
            if j == 0:
                g0 = relu_conv_bn(g0, 'gp{}/'.format(i), num_filter=sum(nf3)+nf1,
                        kernel=(1, 1), pad=(0, 0), use_global_stats=use_global_stats)
            else:
                g0 = group_i
            prefix_name = 'g{}/u{}/'.format(i, j)
            group_i = conv_group(group_i, prefix_name,
                    num_filter_3x3=nf3, num_filter_1x1=nf1, do_proj=True,
                    use_global_stats=use_global_stats)
            group_i = g0 + group_i
        groups.append(group_i)

    nf3_ctx = [(16, 8), (32, 16), (64, 32)]
    nf1_ctx = [8, 16, 32]
    ctx_units = [2, 2, 2]
    for i, (ctx, nf3, nf1) in enumerate(zip(ctx_groups, nf3_ctx, nf1_ctx)):
        group_i = relu_conv_bn(ctx, 'ctxp{}/'.format(i), num_filter=sum(nf3)+nf1,
                kernel=(1, 1), pad=(0, 0), use_global_stats=use_global_stats)
        for j in range(ctx_units[i]):
            prefix_name = 'ctx{}/u{}/'.format(i, j)
            g = conv_group(group_i, prefix_name,
                    num_filter_3x3=nf3, num_filter_1x1=nf1, do_proj=True,
                    use_global_stats=use_global_stats)
            group_i = g + group_i
        ctx_groups[i] = group_i

    up_groups = [[] for _ in groups]
    for i, g in enumerate(up_groups[:3]):
        ss = 2
        for j, ctx in enumerate(ctx_groups[i:], 1):
            u = upsample_feature(ctx, name='up{}{}/'.format(j, i), scale=ss,
                    num_filter_proj=32, num_filter_upsample=32, use_global_stats=use_global_stats)
            up_groups[i].append(u)
            ss *= 2

    return groups, dn_groups, up_groups


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

    bn2 = relu_conv_bn(bn1, '2/',
            num_filter=24, kernel=(3, 3), pad=(1, 1), use_crelu=True,
            use_global_stats=use_global_stats)

    n_curr_ch = 48
    groups, dn_groups, up_groups = prepare_groups(bn2, n_curr_ch, use_global_stats)

    hyper_groups = []
    nf_hyper = [128] * 6
    nf_hyper_proj = [32, 64, 96, 128, 128, 128]

    for i, (g, u, d) in enumerate(zip(groups, up_groups, dn_groups)):
        g = relu_conv_bn(g, 'hyperp{}/'.format(i),
                num_filter=nf_hyper_proj[i], kernel=(1, 1), pad=(0, 0),
                use_global_stats=use_global_stats)
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
    # softmax = mx.sym.SoftmaxOutput(data=pooled_all, label=label, name='softmax')
    fc1 = mx.sym.FullyConnected(pooled_all, num_hidden=4096, name='fc1')
    softmax = mx.sym.SoftmaxOutput(data=fc1, label=label, name='softmax')
    return softmax
