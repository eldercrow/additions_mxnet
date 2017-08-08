import mxnet as mx
import numpy as np
from common import *
from layer.multibox_prior_layer import MultiBoxPriorPython, MultiBoxPriorPythonProp


def relu_conv_bn(data, prefix_name='',
                 num_filter=0, kernel=(3, 3), pad=(0, 0), stride=(1, 1), use_crelu=False,
                 use_global_stats=False, fix_gamma=False, no_bias=True,
                 get_syms=False):
    #
    assert prefix_name != ''
    conv_name = prefix_name + 'conv'
    bn_name = prefix_name + 'bn'
    syms = {}

    relu_ = mx.sym.Activation(data, act_type='relu')
    syms['relu'] = relu_

    conv_ = convolution(relu_, conv_name, num_filter, kernel, pad, stride, no_bias)
    syms['conv'] = conv_

    if use_crelu:
        concat_name = prefix_name + 'concat'
        conv_ = mx.sym.concat(conv_, -conv_, name=concat_name)
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
                    num_filter_1x1=0,
                    n_unit=1,
                    use_global_stats=False,
                    get_syms=False):
    """
    inception unit, only full padding is supported
    """
    # save symbols anyway
    syms = []

    if num_filter_1x1 == 0:
        num_filter_1x1 = num_filter_3x3 * 8

    for n in range(n_unit):
        prefix_name = prefix_group_name + 'u{}/'.format(n)
        sym_n = {}

        bn_, s = relu_conv_bn(data, prefix_name=prefix_name+'init/',
                num_filter=num_filter_3x3, kernel=(1,1), pad=(0,0),
                use_global_stats=use_global_stats, get_syms=True)
        sym_n['init'] = bn_

        incep_layers = [bn_]
        concat_ = bn_
        for ii in range(3):
            bn_, s = relu_conv_bn(
                concat_, prefix_name=prefix_name + '3x3/{}/'.format(ii),
                num_filter=num_filter_3x3, kernel=(3,3), pad=(1,1),
                use_global_stats=use_global_stats, get_syms=True)
            sym_n['unit{}'.format(ii)] = s

            incep_layers.append(bn_)
            concat_ = mx.sym.concat(*incep_layers)

        proj_, s = relu_conv_bn(concat_,
                prefix_name=prefix_name + '1x1/',
                num_filter=num_filter_1x1, kernel=(1,1),
                use_global_stats=use_global_stats, get_syms=True)
        sym_n['concat'] = s

        if num_filter_1x1 != n_curr_ch:
            data, s = relu_conv_bn(data, prefix_name=prefix_name + 'proj/',
                num_filter=num_filter_1x1, kernel=(1,1),
                use_global_stats=use_global_stats, get_syms=True)
            sym_n['proj_data'] = s

        data = proj_ + data
        n_curr_ch = num_filter_1x1
        syms.append(sym_n)

    if get_syms:
        return data, num_filter_1x1, syms
    else:
        return data, num_filter_1x1


def upsample_feature(data,
                     name,
                     scale,
                     num_filter_proj=0,
                     num_filter_upsample=0,
                     use_global_stats=False):
    ''' use subpixel_upsample to upsample a given layer '''
    if num_filter_proj > 0:
        proj = relu_conv_bn(data, prefix_name=name + 'proj/',
            num_filter=num_filter_proj, kernel=(1, 1), pad=(0, 0),
            use_global_stats=use_global_stats)
    else:
        proj = data
    nf = num_filter_upsample * scale * scale
    bn = relu_conv_bn(proj, prefix_name=name + 'conv/',
        num_filter=nf, kernel=(3, 3), pad=(1, 1),
        use_global_stats=use_global_stats)
    return subpixel_upsample(bn, num_filter_upsample, scale, scale)


def get_spotnet(n_classes, patch_size, per_cls_reg, use_global_stats):
    """ main shared conv layers """
    data = mx.sym.Variable(name='data')

    rf_ratio = 4

    conv1 = convolution(data / 128.0, name='1/conv',
        num_filter=16, kernel=(3, 3), pad=(1, 1), no_bias=True)
    concat1 = mx.sym.concat(conv1, -conv1, name='1/concat')
    bn1 = batchnorm(concat1, name='1/bn', use_global_stats=use_global_stats, fix_gamma=False)
    pool1 = pool(bn1)

    bn2 = relu_conv_bn(pool1, prefix_name='2/',
        num_filter=32, kernel=(3, 3), pad=(1, 1), use_crelu=True,
        use_global_stats=use_global_stats)
    pool2 = pool(bn2)
    n_curr_ch = 64

    bn3, n_curr_ch = inception_group(pool2, '3/', n_curr_ch,
            num_filter_3x3=32, num_filter_1x1=128, use_global_stats=use_global_stats)

    nf_3x3 = [48, 64]
    nf_1x1 = [192, 256]
    curr_sz = (2**5) * rf_ratio
    while curr_sz <= patch_size:
        nf_3x3.append(48)
        nf_1x1.append(192)
        curr_sz *= 2

    group_i = bn3
    groups = []
    for i in range(len(nf_3x3)):
        group_i = pool(group_i)
        if i == 2:
            pool_ctx = group_i
            nch_ctx = n_curr_ch
        group_i, n_curr_ch = inception_group(group_i, 'g{}/'.format(i), n_curr_ch,
                num_filter_3x3=nf_3x3[i], num_filter_1x1=nf_1x1[i], n_unit=1,
                use_global_stats=use_global_stats)
        groups.append(group_i)

    group_ctx, _ = inception_group(pool_ctx, 'g_ctx/', nch_ctx,
            num_filter_3x3=32, num_filter_1x1=128, n_unit=1,
            use_global_stats=use_global_stats)

    # build context layers
    upscales = ([4, 2], [2])
    nf_upsamples = ([32, 32], [32])
    nf_proj = 64
    up_layers = (group_ctx, groups[1])
    ctx_layers = []
    for i, (s, u) in enumerate(zip(upscales, nf_upsamples)):
        cl = []
        for j, g in enumerate(up_layers[:len(s)]):
            c = upsample_feature(g, name='ctx{}/{}/'.format(i, j), scale=s[j],
                num_filter_proj=nf_proj, num_filter_upsample=u[j],
                use_global_stats=use_global_stats)
            cl.append(c)
        ctx_layers.append(cl)

    # build multi scale feature layers
    nf_hyper = 384
    nf_hyper_proj = 128
    hyper_layers = []
    # small scale: hyperfeature
    nf_base = [nf_hyper_proj - np.sum(np.array(i)) for i in nf_upsamples]
    for i, g in enumerate(groups[:2]):
        rf = int(2.0**i * 8 * rf_ratio)
        hyper_name = 'hyper{0:03d}/'.format(rf)
        # gather all the upper layers
        g = relu_conv_bn(g, prefix_name='hyperproj/{}/'.format(i),
            num_filter=nf_base[i], kernel=(1, 1), pad=(0, 0),
            use_global_stats=use_global_stats)
        ctxi = [g] + ctx_layers[i]
        ctxi = mx.sym.concat(*ctxi)
        projh = relu_conv_bn(ctxi, prefix_name=hyper_name+'1x1/',
            num_filter=nf_hyper_proj, kernel=(1, 1), pad=(0, 0),
            use_global_stats=use_global_stats)
        hyper_layers.append(projh)

    for i, g in enumerate(groups[2:], 2):
        rf = int(2.0**i * 8 * rf_ratio)
        hyper_name = 'hyper{0:03d}/'.format(rf)
        projh = relu_conv_bn(g, prefix_name=hyper_name+'1x1/',
            num_filter=nf_hyper_proj, kernel=(1, 1), pad=(0, 0),
            use_global_stats=use_global_stats)
        hyper_layers.append(projh)

    # feature layers
    from_layers = []
    strides = []
    sizes = []
    ratios = []
    sz_ratio = np.power(2.0, 1.0 / 2.0)
    for i, (h, g) in enumerate(zip(groups, hyper_layers)):
        st = 2 ** (i + 3)
        rf = st * rf_ratio

        strides.append(st)
        sizes.append([float(rf), float(rf * sz_ratio)])
        ratios.append([1.0, 2.0/3.0, 3.0/2.0, 4.0/9.0, 9.0/4.0])

        hyper_name = 'hyper{0:03d}/'.format(rf)
        resh = relu_conv_bn(g, prefix_name=hyper_name+'res/',
            num_filter=nf_hyper, kernel=(1, 1), pad=(0, 0),
            use_global_stats=use_global_stats)
        convh = relu_conv_bn(h, prefix_name=hyper_name,
            num_filter=nf_hyper, kernel=(3, 3), pad=(1, 1),
            use_global_stats=use_global_stats)
        from_layers.append(convh + resh)

    preds, anchors = multibox_layer_python(from_layers, n_classes,
            sizes=sizes, ratios=ratios, strides=strides, per_cls_reg=per_cls_reg, clip=False)
    return preds, anchors
