import mxnet as mx
from symbol.net_block import *


def prepare_groups(group_i, use_global_stats):
    ''' prepare basic groups '''
    # 96 48 24 12 6 3
    nf_dil = [64, 64, 64, 64, 128, 128]
    dilates = [1, 1, 2, 2, 4, 4]
    groups = []
    for i, (nf, dil) in enumerate(zip(nf_dil, dilates)):
        dilate = (dil, dil)
        pad = dilate
        group_i = relu_conv_bn(group_i, 'gd{}/'.format(i),
                num_filter=nf, kernel=(3, 3), pad=pad, dilate=dilate,
                use_global_stats=use_global_stats)
        groups.append(group_i)

    nf_all = nf_dil * len(dilate) # 512
    group_all = mx.sym.concat(*groups)

    # 24, 48, 96
    groups = []
    g = group_all
    for i in range(3):
        if i > 0:
            g = pool(g)
        g = relu_conv_bn(g, 'g{}/'.format(i),
                num_filter=nf_all, kernel=(3, 3), pad=(1, 1),
                use_global_stats=use_global_stats)
        groups.append(g)

    g = relu_conv_bn(g, 'g3/u0/'.format(i),
            num_filter=nf_all, kernel=(4, 4), pad=(1, 1), stride=(2, 2),
            use_global_stats=use_global_stats)
    g = relu_conv_bn(g, 'g3/u1/'.format(i),
            num_filter=nf_all, kernel=(3, 3), pad=(1, 1),
            use_global_stats=use_global_stats)
    groups.append(g)

    g = relu_conv_bn(g, 'g4/u0/'.format(i),
            num_filter=nf_all, kernel=(4, 4), pad=(1, 1), stride=(2, 2),
            use_global_stats=use_global_stats)
    g = relu_conv_bn(g, 'g4/u1/'.format(i),
            num_filter=nf_all, kernel=(1, 1), pad=(0, 0),
            use_global_stats=use_global_stats)
    groups.append(g)

    g = relu_conv_bn(g, 'g5/'.format(i),
            num_filter=nf_all, kernel=(3, 3), pad=(0, 0),
            use_global_stats=use_global_stats)
    groups.append(g)

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
            num_filter=24, kernel=(3, 3), pad=(1, 1), use_crelu=True,
            use_global_stats=use_global_stats)
    bn2_2 = relu_conv_bn(pool1, '2_2/',
            num_filter=24, kernel=(3, 3), pad=(1, 1),
            use_global_stats=use_global_stats)
    bn2 = mx.sym.concat(bn2_1, bn2_2)
    pool2 = pool(bn2)

    bn3_1 = relu_conv_bn(pool2, '3_1/',
            num_filter=32, kernel=(3, 3), pad=(1, 1), use_crelu=True,
            use_global_stats=use_global_stats)
    bn3_2 = relu_conv_bn(pool2, '3_2/',
            num_filter=64, kernel=(3, 3), pad=(1, 1),
            use_global_stats=use_global_stats)
    bn3 = mx.sym.concat(bn3_1, bn3_2)
    pool3 = pool(bn3)

    groups = prepare_groups(pool3, use_global_stats)

    # nf_group = [192, 192, 192, 192, 144, 144]
    # for i, (g, nf) in enumerate(zip(groups, nf_group)):
    #     g = relu_conv_bn(g, 'gc1x1{}/'.format(i),
    #             num_filter=nf/2, kernel=(1, 1), pad=(0, 0),
    #             use_global_stats=use_global_stats)
    #     g = relu_conv_bn(g, 'gc3x3{}/'.format(i),
    #             num_filter=nf, kernel=(3, 3), pad=(1, 1),
    #             use_global_stats=use_global_stats)
    #     groups[i] = g

    hyper_groups = []
    nf_hyper = [256 for _ in groups] #[192, 192, 192, 192, 192, 192]

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
