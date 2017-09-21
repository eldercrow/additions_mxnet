import mxnet as mx
from symbols.net_block import *
from common.smoothed_softmax_layer import *


def prepare_groups(group_i, use_global_stats):
    ''' prepare basic groups '''
    # 48 24 12 6 3 1
    nf_3x3 = [(128, 64), (256, 128), (384, 192), (128, 64), (128,), ()]
    nf_1x1 = [64, 128, 192, 64, 128, 192]
    n_unit = [2, 3, 3, 1, 1, 1]

    # prepare groups
    n_group = len(nf_1x1)
    groups = [[] for _ in nf_1x1]
    for i, (nf3, nf1) in enumerate(zip(nf_3x3, nf_1x1)):
        kernel = (2, 2) if i < 5 else (3, 3)
        group_i = pool(group_i, kernel=kernel)
        g0 = group_i

        for j in range(n_unit[i]):
            prefix_name = 'g{}/u{}/'.format(i, j)
            group_i = conv_group(group_i, prefix_name,
                    num_filter_3x3=nf3, num_filter_1x1=nf1, do_proj=False,
                    use_global_stats=use_global_stats)

        group_i = proj_add(g0, group_i, 'g{}/res/'.format(i), sum(nf3)+nf1, use_global_stats)
        groups[i].append(group_i)

    groups[0].append( \
            upsample_feature(groups[1][0], name='up10/', scale=2,
                num_filter_proj=64, num_filter_upsample=64, use_global_stats=use_global_stats))
    groups[0].append( \
            upsample_feature(groups[2][0], name='up20/', scale=4,
                num_filter_proj=256, num_filter_upsample=64, use_global_stats=use_global_stats))
    groups[1].append( \
            upsample_feature(groups[2][0], name='up21/', scale=2,
                num_filter_proj=64, num_filter_upsample=64, use_global_stats=use_global_stats))

    for i, g in enumerate(groups[2:-1], 3):
        pad = (1, 1) if i < 4 else (0, 0)
        d = relu_conv_bn(g[0], 'dn{}/'.format(i),
                num_filter=64, kernel=(3, 3), pad=pad, stride=(2, 2),
                use_global_stats=use_global_stats)
        groups[i].append(d)

    return groups


def get_symbol(num_classes=1000, **kwargs):
    '''
    '''
    use_global_stats = kwargs['use_global_stats']

    data = mx.symbol.Variable(name="data")
    label = mx.symbol.Variable(name="softmax_label")

    conv1 = convolution(data, name='1/conv',
        num_filter=24, kernel=(4, 4), pad=(1, 1), stride=(2, 2), no_bias=True)  # 32, 198
    concat1 = mx.sym.concat(conv1, -conv1, name='1/concat')
    bn1 = batchnorm(concat1, name='1/bn', use_global_stats=use_global_stats, fix_gamma=False)

    bn2_1 = relu_conv_bn(bn1, '2_1/',
            num_filter=32, kernel=(4, 4), pad=(1, 1), stride=(2, 2), use_crelu=True,
            use_global_stats=use_global_stats)
    bn2_2 = relu_conv_bn(bn1, '2_2/',
            num_filter=32, kernel=(4, 4), pad=(1, 1), stride=(2, 2),
            use_global_stats=use_global_stats)
    bn2 = mx.sym.concat(bn2_1, bn2_2)

    bn3_1 = relu_conv_bn(bn2, '3_1/',
            num_filter=128, kernel=(3, 3), pad=(1, 1),
            use_global_stats=use_global_stats)
    bn3_2 = relu_conv_bn(bn2, '3_2/',
            num_filter=32, kernel=(3, 3), pad=(1, 1), use_crelu=True,
            use_global_stats=use_global_stats)
    bn3 = mx.sym.concat(bn3_1, bn3_2)

    groups = prepare_groups(bn3, use_global_stats)

    nf_group = [384, 512, 512, 384, 384, 256]
    for i, (g, nf) in enumerate(zip(groups, nf_group)):
        g = mx.sym.concat(*g, name='gc/{}/'.format(i)) if len(g) > 1 else g[0]
        g = relu_conv_bn(g, 'gc1x1/{}/'.format(i),
                num_filter=nf/2, kernel=(1, 1), pad=(0, 0),
                use_global_stats=use_global_stats)
        g = relu_conv_bn(g, 'gc3x3/{}/'.format(i),
                num_filter=nf, kernel=(3, 3), pad=(1, 1),
                use_global_stats=use_global_stats)
        groups[i] = g

    hyper_groups = []
    nf_hyper = [192, 256, 256, 192, 192, 192]

    for i, g in enumerate(groups[:-2]):
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
    ps = 8
    for i, h in enumerate(hyper_groups):
        # hc = mx.sym.concat(h[0], h[1])
        if ps > 1:
            p = mx.sym.Pooling(h[0], kernel=(ps, ps), stride=(ps, ps), pool_type='max')
        else:
            p = h[0]
        ps /= 2
        pooled.append(p)

    pooled_all = mx.sym.flatten(mx.sym.concat(*pooled), name='flatten')
    fc1 = mx.sym.FullyConnected(pooled_all, num_hidden=4096, name='fc1', no_bias=True)
    bn_fc1 = mx.sym.BatchNorm(fc1, use_global_stats=use_global_stats, fix_gamma=False)
    relu_fc1 = mx.sym.Activation(bn_fc1, act_type='relu')
    fc2 = mx.sym.FullyConnected(relu_fc1, num_hidden=num_classes, name='fc2')
    cls_prob = mx.sym.softmax(fc2, name='cls_prob')
    softmax = mx.sym.Custom(fc2, cls_prob, label, op_type='smoothed_softmax_loss', name='softmax',
            th_prob=1e-06, normalization='null')
    # softmax = mx.sym.SoftmaxOutput(data=fc2, label=label, name='softmax')
    return softmax
