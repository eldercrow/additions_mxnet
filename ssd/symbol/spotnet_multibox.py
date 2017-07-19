import mxnet as mx
import numpy as np
from itertools import izip
from common import *
from layer.multibox_prior_layer import MultiBoxPriorPython, MultiBoxPriorPythonProp


def relu_conv_bn(data, prefix_name='',
                 num_filter=0, kernel=(3, 3), pad=(0, 0), stride=(1, 1), dilate=(1, 1), use_crelu=False,
                 use_global_stats=False, fix_gamma=False, no_bias=True,
                 get_syms=False):
    #
    assert prefix_name != ''
    conv_name = prefix_name + 'conv'
    bn_name = prefix_name + 'bn'
    syms = {}
    relu_ = mx.sym.Activation(data, act_type='relu')
    syms['relu'] = relu_

    conv_ = convolution(relu_, conv_name, num_filter, 
            kernel=kernel, pad=pad, stride=stride, dilate=dilate, no_bias=no_bias)
    syms['conv'] = conv_

    if use_crelu:
        conv_ = mx.sym.concat(conv_, -conv_)
        syms['concat'] = conv_

    bn_ = batchnorm(conv_, bn_name, use_global_stats, fix_gamma)
    syms['bn'] = bn_

    if get_syms:
        return bn_, syms
    else:
        return bn_


def inception_group(data,
                    prefix_group_name,
                    n_curr_ch,
                    num_filter_3x3,
                    num_filter_1x1,
                    use_global_stats=False,
                    get_syms=False):
    """
    inception unit, only full padding is supported
    """
    # save symbols anyway
    syms = {}

    prefix_name = prefix_group_name
    dilates = ((1, 1), (1, 1), (2, 2))
    pads = dilates

    incep_layers = []
    concat_ = data
    for ii, (nf3, p, d) in enumerate(izip(num_filter_3x3, pads, dilates)):
        bn_, s = relu_conv_bn(concat_, prefix_name=prefix_name + '3x3/{}/'.format(ii),
            num_filter=nf3, kernel=(3, 3), pad=d, dilate=d,
            use_global_stats=use_global_stats, fix_gamma=True,
            get_syms=True)
        syms['unit{}'.format(ii)] = s
        incep_layers.append(bn_)
        concat_ = bn_ if len(incep_layers) == 1 else mx.sym.concat(*incep_layers)

    concat_, s = relu_conv_bn(concat_, prefix_name=prefix_name + 'concat/',
            num_filter=num_filter_1x1, kernel=(1, 1),
            use_global_stats=use_global_stats, fix_gamma=True,
            get_syms=True)
    syms['concat'] = s

    if num_filter_1x1 != n_curr_ch:
        data, s = relu_conv_bn(data, prefix_name=prefix_name + 'proj/',
            num_filter=num_filter_1x1, kernel=(1, 1),
            use_global_stats=use_global_stats, fix_gamma=True,
            get_syms=True)
        syms['proj_data'] = s

    res_ = concat_ + data

    if get_syms:
        return res_, num_filter_1x1, syms
    else:
        return res_, num_filter_1x1


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


def get_spotnet(n_classes, use_global_stats, patch_size=512):
    """ main shared conv layers """
    data = mx.sym.Variable(name='data')

    conv1 = convolution(data / 128.0, name='1/conv',
        num_filter=12, kernel=(3, 3), pad=(1, 1), no_bias=True)
    concat1 = mx.sym.concat(conv1, -conv1, name='concat1')
    bn1 = batchnorm(concat1, name='1/bn', use_global_stats=use_global_stats, fix_gamma=False)
    pool1 = pool(bn1)

    bn2 = relu_conv_bn(pool1, prefix_name='2/',
        num_filter=24, kernel=(3, 3), pad=(1, 1), use_crelu=True,
        use_global_stats=use_global_stats)
    pool2 = pool(bn2)
    n_curr_ch = 48

    bn3, n_curr_ch = inception_group(pool2, '3/', n_curr_ch,
            num_filter_3x3=(16, 16, 32), num_filter_1x1=128,
            use_global_stats=use_global_stats)

    rf_ratio = 4
    nf_3x3 = [(24, 24, 48), (32, 32, 64), (32, 32, 64)]  # nch: 96, 144, 192
    nf_1x1 = [160, 192, 192]
    strides = [8, 16, 32]

    next_sz = 256
    while next_sz <= patch_size:
        nf_3x3.append((24, 24, 48))
        nf_1x1.append(128)
        strides.append(strides[-1] * 2)
        next_sz *= 2

    sizes = [s * rf_ratio for s in strides]
    ratios = [(1.0, 2.0/3.0, 3.0/2.0, 4.0/9.0, 9.0/4.0)] * len(sizes)

    # basic groups, 20, 40, 80
    group_i = bn3
    groups = []
    for i, (nf3, nf1) in enumerate(izip(nf_3x3, nf_1x1)):
        group_i = pool(group_i)
        group_i, n_curr_ch = inception_group(group_i, 'g{}/'.format(i),
                n_curr_ch, num_filter_3x3=nf3, num_filter_1x1=nf1,
                use_global_stats=use_global_stats,
                get_syms=False)
        groups.append(group_i)

    # build context layers
    upscales = ([4, 2], [2])
    nf_upsamples = ([32, 32], [32])
    nf_proj = 32
    up_layers = (groups[2], groups[1])
    ctx_layers = []
    for i, (s, u) in enumerate(zip(upscales, nf_upsamples)):
        cl = []
        for j, (g, ss, uu) in enumerate(izip(up_layers[:len(s)], s, u)):
            c = upsample_feature(g, name='ctx{}/{}/'.format(i, j), scale=ss,
                        num_filter_proj=nf_proj, num_filter_upsample=uu,
                        use_global_stats=use_global_stats)
            cl.append(c)
        ctx_layers.append(cl)

    # build multi scale feature layers
    from_layers = []
    nf_hyper = 384
    nf_hyper_proj = 128
    # small scale: hyperfeature
    nf_base = [nf_hyper_proj - np.sum(np.array(i)) for i in nf_upsamples]
    for i, (g, cg) in enumerate(izip(groups[:2], ctx_layers)):
        hyper_name = 'hyper{0:03d}/'.format(sizes[i])
        # gather all the upper layers
        g = relu_conv_bn(g, prefix_name=hyper_name+'3x3/',
                num_filter=nf_base[i], kernel=(3, 3), pad=(1, 1),
                use_global_stats=use_global_stats)
        ctxi = [g] + cg
        concat = mx.sym.concat(*(ctxi))

        # create hyper feature layer, append to from_layers
        hyper = relu_conv_bn(concat, prefix_name=hyper_name+'1x1/',
                num_filter=nf_hyper, kernel=(1, 1), pad=(0, 0),
                use_global_stats=use_global_stats)
        from_layers.append(hyper)

    # remaining layers
    for i, g in enumerate(groups[2:], 2):
        hyper_name = 'hyper{0:03d}/'.format(sizes[i])
        projh = relu_conv_bn(g, prefix_name=hyper_name+'3x3/',
                num_filter=nf_hyper_proj, kernel=(3, 3), pad=(1, 1),
                use_global_stats=use_global_stats)
        convh = relu_conv_bn(projh, prefix_name=hyper_name+'1x1/',
                num_filter=nf_hyper, kernel=(1, 1), pad=(0, 0),
                use_global_stats=use_global_stats)
        from_layers.append(convh)

    sz_ratio = np.power(2.0, 1.0 / 4.0)
    sizes_in = [(s / sz_ratio, s * sz_ratio) for s in sizes]
    clip = False

    preds, anchors = multibox_layer_python(from_layers, n_classes,
            sizes=sizes_in, ratios=ratios, strides=strides, per_cls_reg=False, clip=clip)
    return preds, anchors
