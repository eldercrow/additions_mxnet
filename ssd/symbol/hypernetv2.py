import mxnet as mx
from symbol.net_block import *


def prepare_groups(group_i, n_curr_ch, use_global_stats):
    ''' prepare basic groups '''
    feat_sz = [56, 28, 14, 7, 3, 1]
    # feat_sz = [48, 24, 12, 6, 3, 1]
    nf_3x3 = [(48, 32, 32, 16), (72, 48, 48, 24), (96, 48, 48), (64, 32), (96,), ()]
    nf_1x1 = [192, 256, 256, 256, 256, 256]
    nf_init = [64, 96, 128, 128, 128, 256]
    use_init = [False, False, False, True, True, True]

    groups = []
    dn_groups = []
    up_groups = [[] for _ in nf_init[:-1]]

    for i, (nf3, nf1, nfi) in enumerate(zip(nf_3x3, nf_1x1, nf_init)):
        if i == 0 or feat_sz[i-1] % 2 == 0:
            d = relu_conv_bn(group_i, 'dn{}/'.format(i),
                    num_filter=64, kernel=(3, 3), pad=(1, 1), stride=(2, 2),
                    use_global_stats=use_global_stats)
            group_i = pool(group_i)
        else:
            d = relu_conv_bn(group_i, 'dn{}/'.format(i),
                    num_filter=128, kernel=(3, 3), pad=(0, 0), stride=(2, 2),
                    use_global_stats=use_global_stats)
            group_i = pool(group_i, kernel=(3, 3))

        group_i, n_curr_ch = inception_group(group_i, 'g{}/'.format(i), n_curr_ch,
                num_filter_3x3=nf3, num_filter_1x1=nf1, num_filter_init=nfi,
                use_init=use_init[i], use_global_stats=use_global_stats)

        groups.append(group_i)
        dn_groups.append(d)

        if i > 0 and feat_sz[i-1] % 2 == 0:
            s = feat_sz[i-1] / feat_sz[i]
            u = upsample_feature(group_i, 'up{}{}/'.format(i, i-1), scale=s,
                    num_filter_proj=64, num_filter_upsample=64,
                    use_global_stats=use_global_stats)
            up_groups[i-1].append(u)

    up_groups[0].append(\
            upsample_feature(groups[2], 'up20/', scale=4,
                    num_filter_proj=128, num_filter_upsample=64,
                    use_global_stats=use_global_stats))
    up_groups.append([])

    return groups, dn_groups, up_groups


def get_symbol(num_classes=1000, **kwargs):
    '''
    '''
    use_global_stats = kwargs['use_global_stats']

    data = mx.symbol.Variable(name="data")
    label = mx.symbol.Variable(name="label")

    conv1 = convolution(data, name='1/conv',
        num_filter=16, kernel=(4, 4), pad=(1, 1), stride=(2, 2), no_bias=True)  # 32, 198
    concat1 = mx.sym.concat(conv1, -conv1, name='concat1')
    bn1 = batchnorm(concat1, name='1/bn', use_global_stats=use_global_stats, fix_gamma=False)
    pool1 = bn1

    bn2_1 = relu_conv_bn(pool1, '2_1/', num_filter=24, kernel=(3, 3), pad=(1, 1),
            use_crelu=True, use_global_stats=use_global_stats)
    bn2_2 = relu_conv_bn(pool1, '2_2/', num_filter=16, kernel=(3, 3), pad=(1, 1),
            use_global_stats=use_global_stats)
    bn2 = mx.sym.concat(bn2_1, bn2_2)
    pool2 = pool(bn2)

    n_curr_ch = 64
    bn3, n_curr_ch = inception_group(pool2, '3/', n_curr_ch,
            num_filter_3x3=(48, 24, 24), num_filter_1x1=128,
            use_global_stats=use_global_stats)

    groups, dn_groups, up_groups = prepare_groups(bn3, n_curr_ch, use_global_stats)

    hgroups = []
    nf_hyper = (384, 384, 384, 384, 384, 384)
    for i, (g, u, d) in enumerate(zip(groups, up_groups, dn_groups)):
        h = [g, d] + u
        c = mx.sym.concat(*h)
        hc = relu_conv_bn(c, 'hyperc{}/'.format(i),
                num_filter=nf_hyper[i], kernel=(1, 1), pad=(0, 0),
                use_global_stats=use_global_stats)
        hyper = mx.sym.Activation(hc, name='hyper{}'.format(i), act_type='relu')
        hgroups.append(hyper)

    pooled = []
    for i, h in enumerate(hgroups):
        p = mx.sym.Pooling(h, kernel=(2,2), global_pool=True, pool_type='max')
        pooled.append(p)

    pooled_all = mx.sym.flatten(mx.sym.concat(*pooled), name='flatten')
    # softmax = mx.sym.SoftmaxOutput(data=pooled_all, label=label, name='softmax')
    fc1 = mx.sym.FullyConnected(pooled_all, num_hidden=4096, name='fc1')
    softmax = mx.sym.SoftmaxOutput(data=fc1, label=label, name='softmax')
    return softmax
