import mxnet as mx
from symbol.net_block import *


def prepare_groups(group_i, data_shape, n_curr_ch, use_global_stats):
    '''
    '''
    # assume data_shape = 480
    feat_sz = int(data_shape / 4)
    rf = 12

    # prepare filters
    nf_3x3 = [(32, 32), (48, 48, 96), (80, 80, 160), (144, 144, 288)]
    nf_1x1 = [128, 192, 320, 576]

    rf *= 16 # 192
    feat_sz /= 16 # 15

    n_remaining_group = 0
    while rf <= data_shape:
        n_remaining_group += 1
        rf *= 2

    nf3_s7 = (32, 32, 64)
    nf3_s5 = (64, 64)
    nf3_s3 = (128,)
    nf1_sall = 128

    for i in range(n_remaining_group, 0, -1):
        if feat_sz >= 7:
            nf_3x3.append(nf3_s7)
        elif feat_sz >= 5:
            nf_3x3.append(nf3_s5)
        else:
            nf_3x3.append(nf3_s3)
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

        use_crelu = True if i == 0 else False
        # prefix_name = 'g{}/'.format(i)
        prefix_name = 'g{}/'.format(i) if i < 4 else 'gm{}/'.format(n_group-i)
        group_i, n_curr_ch = inception_group(group_i, prefix_name, n_curr_ch,
                num_filter_3x3=nf3, num_filter_1x1=nf1, use_crelu=use_crelu,
                use_global_stats=use_global_stats)
        groups.append(group_i)
        feat_sz /= 2

    return groups, nf_1x1


def predict(groups, nf_proj, use_global_stats, num_class=2):
    '''
    '''
    preds = []
    nf_pred = num_class + 4
    for i, g in enumerate(groups):
        g = relu_conv_bn(g, 'predproj{}/'.format(i),
                num_filter=nf_proj, kernel=(1, 1), pad=(0, 0),
                use_global_stats=use_global_stats)
        relu = mx.sym.Activation(g, act_type='relu')
        p = mx.sym.Convolution(relu, name='pred{}/conv'.format(i),
                num_filter=nf_pred, kernel=(3, 3), pad=(1, 1))
        preds.append(p)
    return preds


def densely_predict(groups, preds, nf_1x1, use_global_stats, num_class=2):
    '''
    '''
    n_group = len(groups)
    up_preds = [[] for _ in groups]

    groups = groups[::-1]
    nf_1x1 = nf_1x1[::-1]
    nf_pred = num_class + 4
    for i, (g, nf1) in enumerate(zip(groups[:-1], nf_1x1[:-1])):
        g = relu_conv_bn(g, 'upproj{}/'.format(i),
                num_filter=nf1/2, kernel=(1, 1), pad=(0, 0),
                use_global_stats=use_global_stats)
        s = 2
        for j in range(i+1, n_group):
            u = upsample_pred(g, name='up{}{}/'.format(i, j), scale=s,
                    num_filter_upsample=nf_pred, use_global_stats=use_global_stats)
            up_preds[j].append(u)
            s *= 2
            nfu /= 2
    up_preds = up_preds[::-1]
    for p, u in zip(preds[:-1], up_preds[:-1]):
        preds[i] = mx.sym.add_n(*([p] + u))
    return preds


def get_symbol(num_classes=1000, **kwargs):
    '''
    '''
    use_global_stats = kwargs['use_global_stats']
    data_shape = kwargs['data_shape']

    data = mx.symbol.Variable(name="data")
    label = mx.symbol.Variable(name="label")

    conv1 = convolution(data, name='1/conv',
        num_filter=16, kernel=(3, 3), pad=(1, 1), stride=(2, 2), no_bias=True)  # 32, 198
    concat1 = mx.sym.concat(conv1, -conv1, name='concat1')
    bn1 = batchnorm(concat1, name='1/bn', use_global_stats=use_global_stats, fix_gamma=False)
    pool1 = bn1

    bn2 = relu_conv_bn(pool1, '2/',
            num_filter=32, kernel=(3, 3), pad=(1, 1), use_crelu=True,
            use_global_stats=use_global_stats)

    n_curr_ch = 64
    groups, nf_1x1 = prepare_groups(bn2, data_shape, n_curr_ch, use_global_stats)

    preds = predict(groups, 128, use_global_stats, 2)
    preds[:4] = densely_predict(groups[:4], nf_1x1[:4], use_global_stats, 2)

    return preds
