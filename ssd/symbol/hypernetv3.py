import mxnet as mx
from symbol.net_block import *


def prepare_groups(group_i, use_global_stats):
    ''' prepare basic groups '''
    # 48 24 12 6 3 1
    nf_3x3 = [(96, 192), (128, 256), (128, 256), (96, 192), (192,), ()]
    nf_1x1 = [96, 128, 128, 96, 96, 192]
    n_unit = [2, 2, 2, 1, 1, 1]

    # prepare groups
    n_group = len(nf_1x1)
    groups = []
    for i, (nf3, nf1) in enumerate(zip(nf_3x3, nf_1x1)):
        kernel = (2, 2) if i < 5 else (3, 3)
        group_i = pool(group_i, kernel=kernel)

        for j in range(n_unit[i]):
            if j == 0:
                nf = nf3[-1] if nf3 else nf1
                g0 = relu_conv_bn(group_i, 'gp{}/'.format(i), num_filter=nf,
                        kernel=(1, 1), pad=(0, 0), use_global_stats=use_global_stats)
            else:
                g0 = group_i
            prefix_name = 'g{}/u{}/'.format(i, j)
            group_i = multiple_conv(group_i, prefix_name,
                    num_filter_3x3=nf3, num_filter_1x1=nf1,
                    use_global_stats=use_global_stats)
            group_i = g0 + group_i

        groups.append([group_i])

    groups[0].append( \
            upsample_feature(groups[1][0], name='up10/', scale=2,
                num_filter_proj=32, num_filter_upsample=32, use_global_stats=use_global_stats))
    groups[0].append( \
            upsample_feature(groups[2][0], name='up20/', scale=4,
                num_filter_proj=32, num_filter_upsample=32, use_global_stats=use_global_stats))

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

    bn2_1 = relu_conv_bn(pool1, '2_1/',
            num_filter=24, kernel=(3, 3), pad=(1, 1), stride=(2, 2), use_crelu=True,
            use_global_stats=use_global_stats)
    bn2_2 = relu_conv_bn(pool1, '2_2/',
            num_filter=16, kernel=(3, 3), pad=(1, 1), stride=(2, 2),
            use_global_stats=use_global_stats)
    bn2 = mx.sym.concat(bn2_1, bn2_2)

    bn3 = multiple_conv(bn2, '3/',
            num_filter_3x3=(64, 128), num_filter_1x1=64,
            use_global_stats=use_global_stats)

    groups = prepare_groups(bn3, use_global_stats)

    nf_group = [512, 512, 512, 384, 384, 256]
    for i, (g, nf) in enumerate(zip(groups, nf_group)):
        g = mx.sym.concat(*g) if len(g) > 1 else g[0]
        g = relu_conv_bn(g, 'gc1x1{}/'.format(i),
                num_filter=nf/2, kernel=(1, 1), pad=(0, 0),
                use_global_stats=use_global_stats)
        g = relu_conv_bn(g, 'gc3x3{}/'.format(i),
                num_filter=nf, kernel=(3, 3), pad=(1, 1),
                use_global_stats=use_global_stats)
        groups[i] = g

    hyper_groups = []
    nf_hyper = [256, 256, 256, 256, 192, 128]

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
