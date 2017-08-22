import mxnet as mx
from symbol.net_block import *
from layer.multibox_prior_layer import *


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

        use_crelu = True if i < 4 else False
        # prefix_name = 'g{}/'.format(i)
        prefix_name = 'g{}/'.format(i) if i < 4 else 'gm{}/'.format(n_group-i)
        group_i, n_curr_ch = conv_group(group_i, prefix_name, n_curr_ch,
                num_filter_3x3=nf3, num_filter_1x1=nf1, use_crelu=use_crelu,
                use_global_stats=use_global_stats)
        groups.append(group_i)
        feat_sz /= 2

    return groups, nf_1x1


def predict(groups, nf_proj, use_global_stats, num_class, sizes, ratios):
    '''
    '''
    preds = []
    for i, g in enumerate(groups):
        nf_pred = (num_class + 4) * len(sizes[i]) * len(ratios[i])
        g = relu_conv_bn(g, 'predproj{}/'.format(i),
                num_filter=nf_proj, kernel=(1, 1), pad=(0, 0),
                use_global_stats=use_global_stats)
        relu = mx.sym.Activation(g, act_type='relu')
        p = mx.sym.Convolution(relu, name='pred{}/conv'.format(i),
                num_filter=nf_pred, kernel=(3, 3), pad=(1, 1))
        preds.append(p)
    return preds


def densely_predict(groups, preds, nf_1x1, use_global_stats, num_class, sizes, ratios):
    '''
    '''
    n_group = len(groups)
    up_preds = [[] for _ in groups]

    groups = groups[::-1]
    nf_1x1 = nf_1x1[::-1]
    for i, (g, nf1) in enumerate(zip(groups[:-1], nf_1x1[:-1])):
        g = relu_conv_bn(g, 'upproj{}/'.format(i),
                num_filter=nf1/2, kernel=(1, 1), pad=(0, 0),
                use_global_stats=use_global_stats)
        s = 2
        for j in range(i+1, n_group):
            nf_pred = (num_class + 4) * len(sizes[j]) * len(ratios[j])
            u = upsample_pred(g, name='up{}{}/'.format(i, j), scale=s,
                    num_filter_upsample=nf_pred, use_global_stats=use_global_stats)
            up_preds[j].append(u)
            s *= 2
    up_preds = up_preds[::-1]
    for p, u in zip(preds[:-1], up_preds[:-1]):
        preds[i] = mx.sym.add_n(*([p] + u))
    return preds


def get_symbol(num_classes, sizes, ratios, steps, **kwargs):
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

    num_classes += 1

    preds = predict(groups, 128, use_global_stats, num_classes, len(sizes), len(ratios))
    preds[:4] = densely_predict(groups[:4], nf_1x1[:4], use_global_stats, \
            num_classes, len(sizes), len(ratios))

    nf_pred = num_classes + 4
    cls_pred_layers = []
    loc_pred_layers = []
    anchor_layers = []

    for p in preds:
        p = mx.sym.transpose(p, (0, 2, 3, 1))
        p = mx.sym.reshape(p, (0, -1, nf_pred))
        cls_pred_layers.append(mx.sym.slice_axis(p, axis=2, begin=0, end=num_classes))
        loc_pred_layers.append(mx.sym.slice_axis(p, axis=2, begin=num_classes, end=None))

    cls_preds = mx.sym.concat(*cls_pred_layers, dim=1)
    cls_preds = mx.sym.transpose(cls_preds, axes=(0, 2, 1), name='multibox_cls_pred')
    loc_preds = mx.sym.concat(*loc_pred_layers, dim=1)
    loc_preds = mx.sym.flatten(loc_preds, name='multibox_loc_preds')

    anchor_boxes = mx.symbol.Custom(*preds, op_type='multibox_prior',
            name='multibox_anchors', sizes=sizes, ratios=ratios, strides=steps)

    return [loc_preds, cls_preds, anchor_boxes]
