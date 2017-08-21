import mxnet as mx
from symbol.net_block import *


def prepare_groups(group_i, data_shape, n_curr_ch, use_global_stats):
    '''
    '''
    # assume data_shape = 480
    feat_sz = int(data_shape / 4)
    rf = 12

    # prepare filters
    nf_3x3 = [(24, 24, 48), (32, 32, 64), (48, 48, 96), (80, 80, 160)]
    nf_1x1 = [0, 0, 0, 0]

    rf *= 16 # 192
    feat_sz /= 16 # 15

    n_remaining_group = 0
    while rf <= data_shape:
        n_remaining_group += 1
        rf *= 2

    nf3_s7 = (48, 48, 96)
    nf3_s5 = (48, 96)
    nf3_s3 = (96,)
    nf1_sall = 96

    for i in range(n_remaining_group, 0, -1):
        if feat_sz >= 7:
            nf_3x3.append(nf3_s7)
            nf_1x1.append(0)
        elif feat_sz >= 5:
            nf_3x3.append(nf3_s5)
            nf_1x1.append(0)
        else:
            nf_3x3.append(nf3_s3)
            nf_1x1.append(96)
        nf_1x1.append(nf1_sall)
        feat_sz /= 2

    # prepare groups
    n_group = len(nf_1x1)
    groups = []
    feat_sz = int(data_shape / 2)
    for i, (nf3, nf1) in enumerate(zip(nf_3x3, nf_1x1)):
        if feat_sz % 2 == 0:
            group_i = pool(group_i)
        else:
            group_i = pool(group_i, kernel=(3, 3))

        # prefix_name = 'g{}/'.format(i)
        prefix_name = 'g{}/'.format(i) if i < 4 else 'gm{}/'.format(n_group-i)
        group_i, n_curr_ch = conv_group(group_i, prefix_name, n_curr_ch,
                num_filter_3x3=nf3, num_filter_1x1=nf1,
                use_global_stats=use_global_stats, use_crelu=True if i < 4 else False)
        groups.append(group_i)
        feat_sz /= 2

    return groups, nf_1x1


def densely_upsample(groups, nf_1x1, use_global_stats):
    '''
    '''
    n_group = len(groups)
    up_groups = [[] for _ in groups]

    groups = groups[::-1]
    nf_1x1 = nf_1x1[::-1]
    for i, (g, nf1) in enumerate(zip(groups[:-1], nf_1x1[:-1])):
        # if up_groups[i]:
        #     g = mx.sym.concat(*([g] + up_groups[i]))
        # g = relu_conv_bn(g, 'upproj{}/'.format(i),
        #         num_filter=nf1/2, kernel=(1, 1), pad=(0, 0),
        #         use_global_stats=use_global_stats)
        s = 2
        nfu = 96
        for j in range(i+1, n_group):
            u = upsample_feature(g, name='up{}{}/'.format(i, j), scale=s,
                    num_filter_proj=nfu*s/2, num_filter_upsample=nfu, use_global_stats=use_global_stats)
            up_groups[j].append(u)
            s *= 2
            nfu /= 2
    return up_groups[::-1]


def get_symbol(num_classes=1000, **kwargs):
    '''
    '''
    use_global_stats = kwargs['use_global_stats']
    data_shape = kwargs['data_shape']

    data = mx.symbol.Variable(name="data")
    label = mx.symbol.Variable(name="label")

    conv1 = convolution(data, name='1/conv',
        num_filter=12, kernel=(3, 3), pad=(1, 1), stride=(1, 1), no_bias=True)  # 32, 198
    concat1 = mx.sym.concat(conv1, -conv1, name='concat1')
    bn1 = batchnorm(concat1, name='1/bn', use_global_stats=use_global_stats, fix_gamma=False)
    pool1 = pool(bn1)

    bn2 = relu_conv_bn(pool1, '2/',
            num_filter=32, kernel=(3, 3), pad=(1, 1), use_crelu=True,
            use_global_stats=use_global_stats)

    n_curr_ch = 64
    groups, nf_1x1 = prepare_groups(bn2, data_shape, n_curr_ch, use_global_stats)

    up_groups = densely_upsample(groups[:4], nf_1x1[:4], use_global_stats)
    for i in range(4, len(groups)):
        up_groups.append([])

    hyper_groups = []
    nf_proj = [24, 48, 96, 192]
    for i in range(4, len(groups)):
        nf_proj.append(192)
    nf_hyper = 192

    for i, (g, u, nfp) in enumerate(zip(groups, up_groups, nf_proj)):
        p = relu_conv_bn(g, 'hyperproj{}/'.format(i),
                num_filter=nfp, kernel=(1, 1), pad=(0, 0),
                use_global_stats=use_global_stats)
        if u:
            p = mx.sym.concat(*([p] + u))
            p = relu_conv_bn(p, 'hyperconcat{}/'.format(i),
                    num_filter=nf_hyper, kernel=(1, 1), pad=(0, 0),
                    use_global_stats=use_global_stats)
        h = mx.sym.Activation(p, name='hyper{}'.format(i), act_type='relu')
        hyper_groups.append(h)

    pooled = []
    for i, h in enumerate(hyper_groups):
        p = mx.sym.Pooling(h, kernel=(2,2), global_pool=True, pool_type='max')
        pooled.append(p)

    pooled_all = mx.sym.flatten(mx.sym.concat(*pooled), name='flatten')
    # softmax = mx.sym.SoftmaxOutput(data=pooled_all, label=label, name='softmax')
    fc1 = mx.sym.FullyConnected(pooled_all, num_hidden=4096, name='fc1')
    softmax = mx.sym.SoftmaxOutput(data=fc1, label=label, name='softmax')
    return softmax
