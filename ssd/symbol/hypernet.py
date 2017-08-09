import mxnet as mx
from symbol.net_block import *


def get_symbol(num_classes=1000, **kwargs):
    '''
    '''
    use_global_stats = kwargs['use_global_stats']

    data = mx.symbol.Variable(name="data")
    label = mx.symbol.Variable(name="label")

    conv1 = convolution(data, name='1/conv',
        num_filter=12, kernel=(3, 3), pad=(1, 1), no_bias=True)  # 32, 198
    concat1 = mx.sym.concat(conv1, -conv1, name='concat1')
    bn1 = batchnorm(concat1, name='1/bn', use_global_stats=use_global_stats, fix_gamma=False)
    pool1 = pool(bn1)

    n_curr_ch = 24

    bn2, n_curr_ch = inception_group(pool1, '2/', n_curr_ch,
            num_filter_3x3=(16, 16), num_filter_1x1=64, use_crelu=True,
            use_global_stats=use_global_stats)
    pool2 = pool(bn2)

    bn3, n_curr_ch = inception_group(pool2, '3/', n_curr_ch,
            num_filter_3x3=(48, 24, 24), num_filter_1x1=96,
            use_global_stats=use_global_stats)

    # 96, 48, 24, 12, 6, 3, 1
    feat_sz = [56, 28, 14, 7, 3, 1]
    nf_3x3 = [(64, 32, 32), (96, 48, 48), (128, 64, 64), (64, 64), (128,), ()]
    nf_1x1 = [128, 192, 256, 128, 128, 128]
    nf_init = [64, 96, 128, 64, 64, 64]

    group_i = bn3
    groups = []
    for i, (nf3, nf1, nfi) in enumerate(zip(nf_3x3, nf_1x1, nf_init)):
        if i > 3:
            group_i = pool(group_i, kernel=(3, 3))
        else:
            group_i = pool(group_i)
        group_i, n_curr_ch = inception_group(group_i, 'g{}/'.format(i), n_curr_ch,
                num_filter_3x3=nf3, num_filter_1x1=nf1, num_filter_init=nfi,
                use_global_stats=use_global_stats)
        groups.append(group_i)

    upgroups = []
    upscales = (2, 2, 2, 3, 3)
    nf_up = (128, 64, 64, 64, 64)
    for i, (g, s, u) in enumerate(zip(groups[1:], upscales, nf_up)):
        u = upsample_feature(g, 'gu{}{}/'.format(i+1, i), scale=s,
                num_filter_proj=64, num_filter_upsample=u,
                use_global_stats=use_global_stats)
        if i == 3:
            u = mx.sym.Crop(u, h_w=(7, 7), center_crop=True)
        upgroups.append([u])
    upgroups[0].append( \
            upsample_feature(groups[2], 'gu20/', scale=4,
                num_filter_proj=64, num_filter_upsample=64,
                use_global_stats=use_global_stats))
    upgroups.append([])

    dgroups = []
    lhs = [bn3] + groups[:-1]
    for i, g in enumerate(lhs):
        if i > 3:
            pad = (0, 0)
        else:
            pad = (1, 1)
        d = relu_conv_bn(g, 'gd{}/'.format(i), 
                num_filter=64, kernel=(3,3), pad=pad, stride=(2,2),
                use_global_stats=use_global_stats)
        # if i == 4:
        #     d = mx.sym.Crop(d, h_w=(3, 3))
        dgroups.append(d)

    hgroups = []
    nf_hyper = (384, 384, 256, 256, 256, 256)
    for i, (g, u, d, nf) in enumerate(zip(groups, upgroups, dgroups, nf_hyper)):
        h = [g] + u + [d]
        c = mx.sym.concat(*h)
        h = relu_conv_bn(c, 'hyper{}/'.format(i),
                num_filter=nf, kernel=(1,1), pad=(0,0),
                use_global_stats=use_global_stats)
        hyper = mx.sym.Activation(h, name='hyper{}'.format(i), act_type='relu')
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
