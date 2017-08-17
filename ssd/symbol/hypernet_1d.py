import mxnet as mx
from symbol.net_block import *


def inception_1d(data, prefix_name, n_curr_ch,
        num_filters, nf_proj, nf_init=0,
        use_global_stats=False):
    '''
    '''
    if nf_init > 0:
        bn_ = relu_conv_bn(data, prefix_name=prefix_name+'init/',
                num_filter=nf_init, kernel=(1, 1), pad=(0, 0),
                use_global_stats=use_global_stats)
    else:
        bn_ = data

    # horizontal
    unit_h = []
    for i, nf in enumerate(num_filters):
        sz = i * 2 + 3
        name = prefix_name + '1x{}/0/'.format(sz)
        if nf > 0:
            h = relu_conv_bn(bn_, prefix_name=name,
                    num_filter=nf, kernel=(1, sz), pad=(0, i+1),
                    use_global_stats=use_global_stats)
            unit_h.append(h)
    if len(unit_h) > 0:
        data_h = mx.sym.concat(*unit_h)

    # vertical
    unit_v = []
    for i, nf in enumerate(num_filters):
        sz = i * 2 + 3
        name = prefix_name + '{}x1/0/'.format(sz)
        if nf > 0:
            v = relu_conv_bn(bn_, prefix_name=name,
                    num_filter=nf, kernel=(sz, 1), pad=(i+1, 0),
                    use_global_stats=use_global_stats)
            unit_v.append(v)
    if len(unit_v) > 0:
        data_v = mx.sym.concat(*unit_v)

    # horizontal
    unit_all = []
    for i, nf in enumerate(num_filters):
        sz = i * 2 + 3
        name = prefix_name + '1x{}/1/'.format(sz)
        if nf > 0:
            h = relu_conv_bn(data_v, prefix_name=name,
                    num_filter=nf, kernel=(1, sz), pad=(0, i+1),
                    use_global_stats=use_global_stats)
            unit_all.append(h)

    # vertical
    for i, nf in enumerate(num_filters):
        sz = i * 2 + 3
        name = prefix_name + '{}x1/1/'.format(sz)
        if nf > 0:
            v = relu_conv_bn(data_h, prefix_name=name,
                    num_filter=nf, kernel=(sz, 1), pad=(i+1, 0),
                    use_global_stats=use_global_stats)
            unit_all.append(v)

    if len(unit_all) > 0:
        concat = mx.sym.concat(*unit_all)
    else:
        concat = bn_
    concat = relu_conv_bn(concat, prefix_name=prefix_name+'concat/',
            num_filter=nf_proj, kernel=(1, 1), pad=(0, 0),
            use_global_stats=use_global_stats)

    if nf_proj != n_curr_ch:
        data = relu_conv_bn(data, prefix_name=prefix_name+'proj/',
            num_filter=nf_proj, kernel=(1, 1),
            use_global_stats=use_global_stats)

    return concat + data, nf_proj


def get_symbol(num_classes=1000, **kwargs):
    '''
    '''
    use_global_stats = kwargs['use_global_stats']

    data = mx.symbol.Variable(name="data")
    label = mx.symbol.Variable(name="label")

    conv1 = convolution(data, name='1/conv',
        num_filter=16, kernel=(3, 3), pad=(1, 1), no_bias=True)  # 32, 198
    concat1 = mx.sym.concat(conv1, -conv1, name='concat1')
    bn1 = batchnorm(concat1, name='1/bn', use_global_stats=use_global_stats, fix_gamma=False)
    pool1 = pool(bn1)

    n_curr_ch = 32

    bn2, n_curr_ch = inception_group(pool1, '2/', n_curr_ch,
            num_filter_3x3=(32, 32), num_filter_1x1=64,
            use_global_stats=use_global_stats)
    pool2 = pool(bn2)

    bn3, n_curr_ch = inception_group(pool2, '3/', n_curr_ch,
            num_filter_3x3=(64, 32, 32), num_filter_1x1=128,
            use_global_stats=use_global_stats)

    # feat_sz = [56, 28, 14, 7, 3, 1]
    nf_3x3 = [(96, 48, 48), (128, 64, 64), (128, 64, 64), (128, 128), (128,), ()]
    nf_1x1 = [192, 256, 256, 256, 128, 128]
    nf_init = [96, 128, 128, 128, 64, 64]

    group_i = bn3
    groups = []
    for i, (nf3, nf1, nfi) in enumerate(zip(nf_3x3, nf_1x1, nf_init)):
        if i > 4:
            group_i = pool(group_i, kernel=(3, 3))
        else:
            group_i = pool(group_i)
        group_i, n_curr_ch = inception_group(group_i, 'g{}/'.format(i), n_curr_ch,
                num_filter_3x3=nf3, num_filter_1x1=nf1, num_filter_init=nfi,
                use_global_stats=use_global_stats)
        groups.append(group_i)

    upgroups = []
    upscales = (2, 2, 2, 2, 3)
    nf_up = (64, 64, 64, 64, 64)
    for i, (g, s, u) in enumerate(zip(groups[1:], upscales, nf_up)):
        u = upsample_feature(g, 'gu{}{}/'.format(i+1, i), scale=s,
                num_filter_proj=64, num_filter_upsample=u,
                use_global_stats=use_global_stats)
        # if i == 3:
        #     u = mx.sym.Crop(u, h_w=(7, 7), center_crop=True)
        upgroups.append([u])
    upgroups[0].append( \
            upsample_feature(groups[2], 'gu20/', scale=4,
                num_filter_proj=64, num_filter_upsample=64,
                use_global_stats=use_global_stats))
    upgroups.append([])

    dgroups = []
    lhs = [bn3] + groups[:-1]
    for i, g in enumerate(lhs):
        if i > 4:
            pad = (0, 0)
            kernel = (3, 3)
        else:
            pad = (1, 1)
            kernel = (3, 3)
        d = relu_conv_bn(g, 'gd{}/'.format(i),
                num_filter=64, kernel=kernel, pad=pad, stride=(2,2),
                use_global_stats=use_global_stats)
        dgroups.append(d)

    hgroups = []
    nf_hyper = (384, 384, 384, 384, 256, 256)
    nf_357 = [(48, 48, 48), (48, 48, 48), (48, 48, 48), (64, 64), (64,), ()]
    nf_hyper_proj = (192, 192, 192, 192, 128, 64)
    for i, (g, u, d, nf, nf_p) in enumerate(zip(groups, upgroups, dgroups, nf_hyper, nf_hyper_proj)):
        h = [g] + u + [d]
        c = mx.sym.concat(*h)
        hc = relu_conv_bn(c, 'hyperc{}/'.format(i),
                num_filter=nf, kernel=(1,1), pad=(0,0),
                use_global_stats=use_global_stats)
        h, _ = inception_1d(hc, 'hyper1d{}/'.format(i), n_curr_ch=nf,
                num_filters=nf_357[i], nf_proj=nf, nf_init=nf_p,
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
