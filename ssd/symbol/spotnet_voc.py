import mxnet as mx
import numpy as np
from layer.multibox_prior_layer import *


def convolution(data, name, num_filter, kernel, pad, stride=(1,1), no_bias=False, lr_mult=1.0):
    ''' convolution with lr_mult and wd_mult '''
    w = mx.sym.var(name+'_weight', lr_mult=lr_mult, wd_mult=lr_mult)
    b = None
    if no_bias == False:
        b = mx.sym.var(name+'_bias', lr_mult=lr_mult*2.0, wd_mult=0.0)
    conv = mx.sym.Convolution(data, weight=w, bias=b, name=name, num_filter=num_filter,
            kernel=kernel, pad=pad, stride=stride, no_bias=no_bias)
    return conv


def fullyconnected(data, name, num_hidden, no_bias=False, lr_mult=1.0):
    ''' convolution with lr_mult and wd_mult '''
    w = mx.sym.var(name+'_weight', lr_mult=lr_mult, wd_mult=lr_mult)
    b = None
    if no_bias == False:
        b = mx.sym.var(name+'_bias', lr_mult=lr_mult*2.0, wd_mult=0.0)
    fc = mx.sym.FullyConnected(data, weight=w, bias=b, name=name, num_hidden=num_hidden, no_bias=no_bias)
    return fc


def batchnorm(data, name, use_global_stats, fix_gamma=False, lr_mult=1.0):
    ''' batch norm with lr_mult and wd_mult '''
    g = mx.sym.var(name+'_gamma', lr_mult=lr_mult, wd_mult=0.0)
    b = mx.sym.var(name+'_beta', lr_mult=lr_mult, wd_mult=0.0)
    bn = mx.sym.BatchNorm(data, gamma=g, beta=b, name=name,
            use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    return bn


def pool(data, name=None, kernel=(2, 2), stride=(2, 2), pool_type='max'):
    return mx.sym.Pooling(data, name=name, kernel=kernel, stride=stride, pool_type=pool_type)


def subpixel_upsample(data, ch, c, r):
    if r == 1 and c == 1:
        return data
    X = mx.sym.reshape(data=data, shape=(-3, 0, 0))  # (bsize*ch*r*c, a, b)
    X = mx.sym.reshape(
        data=X, shape=(-4, -1, r * c, 0, 0))  # (bsize*ch, r*c, a, b)
    X = mx.sym.transpose(data=X, axes=(0, 3, 2, 1))  # (bsize*ch, b, a, r*c)
    X = mx.sym.reshape(data=X, shape=(0, 0, -1, c))  # (bsize*ch, b, a*r, c)
    X = mx.sym.transpose(data=X, axes=(0, 2, 1, 3))  # (bsize*ch, a*r, b, c)
    X = mx.sym.reshape(
        data=X, shape=(-4, -1, ch, 0, -3))  # (bsize, ch, a*r, b*c)
    return X


def relu_conv_bn(data, prefix_name='',
                 num_filter=0, kernel=(3, 3), pad=(0, 0), stride=(1, 1), use_crelu=False,
                 use_global_stats=False, fix_gamma=False, no_bias=True):
    #
    assert prefix_name != ''
    conv_name = prefix_name + 'conv'
    bn_name = prefix_name + 'bn'

    relu_ = mx.sym.Activation(data, act_type='relu')
    conv_ = convolution(relu_, conv_name, num_filter, kernel, pad, stride, no_bias)
    if use_crelu:
        concat_name = prefix_name + 'concat'
        conv_ = mx.sym.concat(conv_, -conv_, name=concat_name)

    bn_ = batchnorm(conv_, bn_name, use_global_stats, fix_gamma)
    return bn_


def multibox_layer_python(from_layers, num_classes, sizes, ratios, strides, per_cls_reg=False, clip=False):
    ''' multibox layer '''
    # parameter check
    assert len(from_layers) > 0, "from_layers must not be empty list"
    assert num_classes > 1, "num_classes {} must be larger than 1".format(num_classes)
    assert len(ratios) == len(from_layers), "ratios and from_layers must have same length"
    assert len(sizes) == len(from_layers), "sizes and from_layers must have same length"

    pred_layers = []
    anchor_layers = []

    num_reg = 4 * num_classes if per_cls_reg else 4

    for k, from_layer in enumerate(from_layers):
        from_name = from_layer.name
        num_anchors = len(sizes[k]) * len(ratios[k])
        num_loc_pred = num_anchors * num_reg
        num_cls_pred = num_anchors * num_classes
        num_filter = num_loc_pred + num_cls_pred

        pred_conv = convolution(from_layer, name='{}_pred/conv'.format(from_name),
                num_filter=num_filter, kernel=(3, 3), pad=(1, 1))
        pred_conv = mx.sym.transpose(pred_conv, axes=(0, 2, 3, 1))  # (n h w ac), a=num_anchors
        pred_conv = mx.sym.reshape(pred_conv, shape=(0, -1, num_classes + num_reg))
        # pred_conv = mx.sym.reshape(pred_conv, shape=(0, -3, -4, num_anchors, -1))  # (n h*w a c)
        # pred_conv = mx.sym.reshape(pred_conv, shape=(0, -3, -1))  # (n h*w*a c)
        pred_layers.append(pred_conv)

    anchors = mx.sym.Custom(*from_layers, op_type='multibox_prior_python',
        sizes=sizes, ratios=ratios, strides=strides, clip=int(clip))
    preds = mx.sym.concat(*pred_layers, num_args=len(pred_layers), dim=1)
    return [preds, anchors]


def inception_group(data,
                    prefix_group_name,
                    n_curr_ch,
                    num_filter_3x3,
                    num_filter_1x1,
                    n_unit=1,
                    use_global_stats=False):
    """
    inception unit, only full padding is supported
    """
    for n in range(n_unit):
        prefix_name = prefix_group_name + 'u{}/'.format(n)
        incep_layers = []
        bn_ = relu_conv_bn(data, prefix_name=prefix_name + 'init/',
            num_filter=num_filter_3x3, kernel=(1,1), pad=(0,0),
            use_global_stats=use_global_stats)
        incep_layers.append(bn_)
        concat_ = bn_

        for ii in range(3):
            bn_ = relu_conv_bn(concat_, prefix_name=prefix_name + '3x3/{}/'.format(ii),
                num_filter=num_filter_3x3, kernel=(3,3), pad=(1,1),
                use_global_stats=use_global_stats)
            incep_layers.append(bn_)
            concat_ = mx.sym.concat(*incep_layers)

        proj_ = relu_conv_bn(concat_, prefix_name=prefix_name + '1x1/',
                num_filter=num_filter_1x1, kernel=(1,1), pad=(0,0),
                use_global_stats=use_global_stats)

        if num_filter_1x1 != n_curr_ch:
            data = relu_conv_bn(data, prefix_name=prefix_name + 'proj/',
                num_filter=num_filter_1x1, kernel=(1,1), use_global_stats=use_global_stats)

        data = data + proj_
    return data, num_filter_1x1


def upsample_feature(data,
                     name,
                     scale,
                     num_filter_proj=0,
                     num_filter_upsample=0,
                     use_global_stats=False):
    ''' use subpixel_upsample to upsample a given layer '''
    if num_filter_proj > 0:
        proj = relu_conv_bn(
            data,
            prefix_name=name + 'proj/',
            num_filter=num_filter_proj,
            kernel=(1, 1),
            pad=(0, 0),
            use_global_stats=use_global_stats)
    else:
        proj = data
    nf = num_filter_upsample * scale * scale
    bn = relu_conv_bn(
        proj,
        prefix_name=name + 'conv/',
        num_filter=nf,
        kernel=(3, 3),
        pad=(1, 1),
        use_global_stats=use_global_stats)
    return subpixel_upsample(bn, num_filter_upsample, scale, scale)


def get_spotnet(n_classes, patch_size, per_cls_reg, use_global_stats):
    """ main shared conv layers """
    data = mx.sym.Variable(name='data')

    conv1 = convolution(data / 128.0, name='1/conv',
        num_filter=16, kernel=(3, 3), pad=(1, 1), no_bias=True)  # 32, 198
    concat1 = mx.sym.concat(conv1, -conv1, name='concat1')
    bn1 = batchnorm(concat1, name='1/bn', use_global_stats=use_global_stats, fix_gamma=False)
    pool1 = pool(bn1)

    bn2 = relu_conv_bn(pool1, prefix_name='2/',
        num_filter=32, kernel=(3, 3), pad=(1, 1), use_crelu=True,
        use_global_stats=use_global_stats)
    pool2 = pool(bn2)

    n_curr_ch = 64
    bn3, n_curr_ch = inception_group(
            pool2, '3/', n_curr_ch, n_unit=1, 
            num_filter_3x3=16, num_filter_1x1=16*8, 
            use_global_stats=use_global_stats)

    nf_3x3 = [24, 32, 32] # rf 16, 32, 64, 128
    nf_1x1 = [i*8 for i in nf_3x3]
    curr_sz = 128
    while curr_sz < patch_size:
        nf_3x3.append(24)
        nf_1x1.append(24*8)
        curr_sz *= 2

    # basic groups, 32, 64
    group_i = bn3
    groups = []
    for i in range(len(nf_3x3)):
        group_i = pool(group_i)
        group_i, n_curr_ch = inception_group(
            group_i, 'g{}/'.format(i), n_curr_ch, n_unit=1,
            num_filter_3x3=nf_3x3[i], num_filter_1x1=nf_1x1[i],
            use_global_stats=use_global_stats)
        groups.append(group_i)

    # build context layers
    upscales = [[4, 2], [2]]
    nf_upsamples = [[32, 64], [64]]
    nf_proj = 32
    up_layers = [groups[2], groups[1]]
    ctx_layers = []
    for i, (s, u) in enumerate(zip(upscales, nf_upsamples)):
        cl = []
        # context layers for 32, 64
        for j, g in enumerate(up_layers[:len(s)]):
            c = upsample_feature(g, name='ctx{}/{}'.format(i, j),
                    scale=s[j], num_filter_proj=nf_proj, num_filter_upsample=u[j],
                    use_global_stats=use_global_stats)
            cl.append(c)
        ctx_layers.append(cl)

    # build multi scale feature layers
    from_layers = []
    nf_hyper = 384
    # small scale: hyperfeature
    hyper_names = ['hyper032/', 'hyper064/']
    nf_base = [nf_hyper - np.sum(np.array(i)) for i in nf_upsamples]
    for i, g in enumerate(groups[:2]):
        # gather all the upper layers
        g = relu_conv_bn(g, prefix_name='hyperproj/{}/'.format(i),
            num_filter=nf_base[i], kernel=(1, 1), pad=(0, 0),
            use_global_stats=use_global_stats)
        ctxi = ctx_layers[i] + [g]
        concat = mx.sym.concat(*(ctxi))
        convh = relu_conv_bn(concat, prefix_name=hyper_names[i],
            num_filter=nf_hyper, kernel=(1, 1), pad=(0, 0),
            use_global_stats=use_global_stats)
        from_layers.append(convh)

    # remaining layers, bigger than 64
    for i, g in enumerate(groups[2:], 2):
        rf = int((2.0**i) * 32.0)
        prefix_name = 'hyper{}/'.format(rf)
        convh = relu_conv_bn(g, prefix_name='hyper{}/'.format(rf),
            num_filter=nf_hyper, kernel=(1, 1), pad=(0, 0),
            use_global_stats=use_global_stats)
        from_layers.append(convh)

    n_from_layers = len(from_layers)
    strides = []
    sizes = []
    sz_ratio = np.power(2.0, 1.0 / 4.0)
    for i in range(n_from_layers):
        st = 2 ** (i + 3)
        sz = st * 4.0
        strides.append(st)
        sizes.append([sz / sz_ratio, sz * sz_ratio])
    ratios = [[1.0, 2.0/3.0, 3.0/2.0, 4.0/9.0, 9.0/4.0]] * len(sizes)
    clip = False

    preds, anchors = multibox_layer_python(from_layers, n_classes,
            sizes=sizes, ratios=ratios, strides=strides, per_cls_reg=per_cls_reg, clip=False)
    return preds, anchors
