import mxnet as mx
from symbol.net_block import *


def prepare_groups(data, use_global_stats):
    ''' prepare basic groups '''
    # 48 24 12 6 3 1
    nf_3x3 = [(128, 64), (160, 80), (192, 96), (128, 64), (96,), ()]
    nf_1x1 = [64, 80, 96, 64, 96, 192]
    n_unit = [2, 3, 3, 1, 1, 1]

    # prepare groups
    n_group = len(nf_1x1)
    groups = []
    group_i = data
    for i, (nf3, nf1) in enumerate(zip(nf_3x3, nf_1x1)):
        kernel = (2, 2) if i < 5 else (3, 3)
        group_i = pool(group_i, kernel=kernel)
        g0 = group_i

        for j in range(n_unit[i]):
            prefix_name = 'g{}/u{}/'.format(i, j)
            group_i = conv_dilate_group(group_i, prefix_name,
                    num_filter_3x3=nf3, num_filter_1x1=nf1, do_proj=False,
                    use_global_stats=use_global_stats)

        group_i = proj_add(g0, group_i, 'g{}/res/'.format(i), sum(nf3)+nf1, use_global_stats)
        groups.append(group_i)

    # context feature
    ctx, ctx2 = prepare_context(groups[0], use_global_stats)
    groups[0] = mx.sym.concat(groups[0], ctx)
    groups[1] = mx.sym.concat(groups[1], ctx2)

    return groups


def prepare_context(data, use_global_stats):
    ''' prepare context info '''
    nf_3x3 = (32, 64, 128)
    nf_1x1 = (32, 32, 64)
    dilate = (1, 2, 4)
    n_unit = (1, 2, 2)

    ctx = data
    for i, (nf3, nf1, dil) in enumerate(zip(nf_3x3, nf_1x1, dilate)):
        for j in range(n_unit[i]):
            ctx = relu_conv_bn(ctx, 'ctx1x1{}/u{}/'.format(i, j),
                    num_filter=nf1, kernel=(1, 1), pad=(0, 0),
                    use_global_stats=use_global_stats)
            ctx = relu_conv_bn(ctx, 'ctx3x3{}/u{}/'.format(i, j),
                    num_filter=nf3, kernel=(3, 3), pad=(dil, dil), dilate=(dil, dil),
                    use_global_stats=use_global_stats)

    ctx2 = relu_conv_bn(ctx, 'ctx2/1x1/',
            num_filter=64, kernel=(1, 1), pad=(0, 0),
            use_global_stats=use_global_stats)
    ctx2 = relu_conv_bn(ctx2, 'ctx2/3x3/',
            num_filter=64, kernel=(3, 3), pad=(1, 1), stride=(2, 2),
            use_global_stats=use_global_stats)
    return (ctx, ctx2)


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

    bn2_1 = relu_conv_bn(bn1, '2_1/',
            num_filter=32, kernel=(3, 3), pad=(1, 1), use_crelu=True,
            use_global_stats=use_global_stats)
    bn2_2 = relu_conv_bn(bn1, '2_2/',
            num_filter=32, kernel=(3, 3), pad=(1, 1),
            use_global_stats=use_global_stats)
    bn2 = mx.sym.concat(bn2_1, bn2_2)
    pool2 = pool(bn2)

    bn3 = conv_group(pool2, '3/',
            num_filter_3x3=(96, 48), num_filter_1x1=48, do_proj=False,
            use_global_stats=use_global_stats)
    bn3 = proj_add(pool2, bn3, '3/res/', 192, use_global_stats)

    groups = prepare_groups(bn3, use_global_stats)

    nf_group = [384, 512, 512, 384, 384, 256]
    for i, (g, nf) in enumerate(zip(groups, nf_group)):
        # g = mx.sym.concat(*g) if len(g) > 1 else g[0]
        g = relu_conv_bn(g, 'gc1x1/{}/'.format(i),
                num_filter=nf/2, kernel=(1, 1), pad=(0, 0),
                use_global_stats=use_global_stats)
        g = relu_conv_bn(g, 'gc3x3/{}/'.format(i),
                num_filter=nf, kernel=(3, 3), pad=(1, 1),
                use_global_stats=use_global_stats)
        groups[i] = g

    hyper_groups = []
    nf_hyper = [192, 256, 256, 192, 192, 128]

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
    fc1 = mx.sym.FullyConnected(pooled_all, num_hidden=4096, name='fc1')
    softmax = mx.sym.SoftmaxOutput(data=fc1, label=label, name='softmax')
    return softmax
