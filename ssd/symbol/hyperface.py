import mxnet as mx
from symbol.net_block import *


def prepare_groups(group_i, use_global_stats):
    ''' prepare basic groups '''
    # 96 48 24 12 6 3
    nf_3x3 = [(48, 24), (64, 32), (80, 40), (96, 48), (64, 32), (64,)]
    nf_1x1 = [24, 32, 40, 48, 32, 64]
    n_unit = [2, 2, 2, 2, 1, 1]

    # prepare groups
    n_group = len(nf_1x1)
    groups = []
    for i, (nf3, nf1) in enumerate(zip(nf_3x3, nf_1x1)):
        group_i = pool(group_i)
        g0 = group_i

        for j in range(n_unit[i]):
            prefix_name = 'g{}/u{}/'.format(i, j)
            group_i = conv_group(group_i, prefix_name,
                    num_filter_3x3=nf3, num_filter_1x1=nf1, do_proj=False,
                    use_global_stats=use_global_stats)

        group_i = proj_add(g0, group_i, 'g{}/res/'.format(i), sum(nf3)+nf1, use_global_stats)
        groups.append([group_i])

    # context feature
    ctx1, ctx2, ctx3 = prepare_context(groups[0], use_global_stats)
    groups[0] = mx.sym.concat(groups[0], ctx1)
    groups[1] = mx.sym.concat(groups[1], ctx2)
    groups[2] = mx.sym.concat(groups[1], ctx3)

    return groups


def prepare_context(data, use_global_stats):
    ''' prepare context info '''
    nf_3x3 = (16, 32, 64, 96)
    nf_1x1 = (16, 16, 32, 48)
    dilate = (1, 2, 4, 8)
    n_unit = (1, 1, 1, 1)

    ctx = data
    for i, (nf3, nf1, dil) in enumerate(zip(nf_3x3, nf_1x1, dilate)):
        for j in range(n_unit[i]):
            ctx = relu_conv_bn(ctx, 'ctxdil{}/1x1/u{}/'.format(i, j),
                    num_filter=nf1, kernel=(1, 1), pad=(0, 0),
                    use_global_stats=use_global_stats)
            ctx = relu_conv_bn(ctx, 'ctxdil{}/3x3/u{}/'.format(i, j),
                    num_filter=nf3, kernel=(3, 3), pad=(dil, dil), dilate=(dil, dil),
                    use_global_stats=use_global_stats)

    ctx_all = [ctx]
    nf_ctx = (64, 32)
    for i, nf in enumerate(nf_ctx, 1):
        ctx = relu_conv_bn(ctx, 'ctx{}/1x1/'.format(i),
                num_filter=nf, kernel=(1, 1), pad=(0, 0),
                use_global_stats=use_global_stats)
        ctx = relu_conv_bn(ctx, 'ctx{}/3x3/'.format(i),
                num_filter=nf, kernel=(3, 3), pad=(1, 1), stride=(2, 2),
                use_global_stats=use_global_stats)
        ctx_all.append(ctx)
    return ctx_all


def get_symbol(num_classes=1000, **kwargs):
    '''
    '''
    use_global_stats = kwargs['use_global_stats']

    data = mx.symbol.Variable(name="data")
    label = mx.symbol.Variable(name="label")

    conv1 = convolution(data, name='1/conv',
        num_filter=12, kernel=(4, 4), pad=(1, 1), stride=(2, 2), no_bias=True)  # 32, 198
    concat1 = mx.sym.concat(conv1, -conv1, name='1/concat')
    bn1 = batchnorm(concat1, name='1/bn', use_global_stats=use_global_stats, fix_gamma=False)

    bn2_1 = relu_conv_bn(bn1, '2_1/',
            num_filter=16, kernel=(3, 3), pad=(1, 1), use_crelu=True,
            use_global_stats=use_global_stats)
    bn2_2 = relu_conv_bn(bn1, '2_2/',
            num_filter=16, kernel=(3, 3), pad=(1, 1),
            use_global_stats=use_global_stats)
    bn2 = mx.sym.concat(bn2_1, bn2_2)

    groups = prepare_groups(bn2, use_global_stats)

    nf_group = [192, 192, 192, 192, 192, 192]
    for i, (g, nf) in enumerate(zip(groups, nf_group)):
        g = relu_conv_bn(g, 'gc1x1{}/'.format(i),
                num_filter=nf/2, kernel=(1, 1), pad=(0, 0),
                use_global_stats=use_global_stats)
        g = relu_conv_bn(g, 'gc3x3{}/'.format(i),
                num_filter=nf, kernel=(3, 3), pad=(1, 1),
                use_global_stats=use_global_stats)
        groups[i] = g

    hyper_groups = []
    nf_hyper = [96, 96, 96, 96, 96, 96]

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
