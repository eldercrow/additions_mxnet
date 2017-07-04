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
                    use_crelu=False,
                    use_global_stats=False,
                    get_syms=False):
    """
    inception unit, only full padding is supported
    """
    # save symbols anyway
    syms = {}

    prefix_name = prefix_group_name
    incep_layers = []
    bn_, s = relu_conv_bn(data, prefix_name=prefix_name+'1x1/init/',
            num_filter=num_filter_3x3[0], kernel=(1,1), pad=(0,0),
            use_global_stats=use_global_stats, get_syms=True)
    syms['init'] = bn_
    # bn_ = data
    # nch = n_curr_ch
    num_filter_incep = 0
    for ii in range(3):
        bn_, s = relu_conv_bn(
            bn_,
            prefix_name=prefix_name + '3x3/{}/'.format(ii),
            num_filter=num_filter_3x3[ii],
            kernel=(3, 3),
            pad=(1, 1),
            use_crelu=use_crelu,
            use_global_stats=use_global_stats,
            get_syms=True)
        syms['unit{}'.format(ii)] = s
        incep_layers.append(bn_)
        nch = num_filter_3x3[ii]
        if use_crelu:
            nch *= 2
        num_filter_incep += nch

    concat_ = mx.sym.concat(*incep_layers)
    concat_, s = relu_conv_bn(concat_,
            prefix_name=prefix_name + 'concat/',
            num_filter=num_filter_incep,
            kernel=(1, 1),
            use_global_stats=use_global_stats,
            get_syms=True)
    syms['concat'] = s

    if num_filter_incep != n_curr_ch:
        # data = mx.sym.Convolution(data, name=prefix_name + 'proj/conv',
        #         num_filter=num_filter_incep, kernel=(1,1))
        # syms['proj_data'] = data
        data, s = relu_conv_bn(
            data,
            prefix_name=prefix_name + 'proj/',
            num_filter=num_filter_incep,
            kernel=(1, 1),
            use_global_stats=use_global_stats,
            get_syms=True)
        syms['proj_data'] = s

    res_ = concat_ + data

    if get_syms:
        return res_, num_filter_incep, syms
    else:
        return res_, num_filter_incep


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

    nf_3x3 = ((32, 16, 16), (48, 24, 24), (64, 32, 32))  # nch: 96, 144, 192

    # basic groups, 12, 24, 48
    group_i = bn2
    groups = []
    n_curr_ch = 48
    for i in range(len(nf_3x3)):
        group_i = pool(group_i)
        group_i, n_curr_ch = inception_group(
            group_i,
            'g{}/u0/'.format(i),
            n_curr_ch,
            num_filter_3x3=nf_3x3[i],
            use_crelu=True,
            use_global_stats=use_global_stats,
            get_syms=False)
        groups.append(group_i)

    # 96 and more
    curr_sz = 48
    i = 3
    while curr_sz < patch_size:
        group_i = pool(group_i)
        if i == 3:
            pool5 = group_i
        group_i, n_curr_ch = inception_group(
            group_i,
            'g{}/u0/'.format(i),
            n_curr_ch,
            num_filter_3x3=nf_3x3[-1],
            use_global_stats=use_global_stats,
            get_syms=False)
        groups.append(group_i)
        curr_sz *= 2
        i += 1

    group_ctx, _ = inception_group(
            pool5,
            'g_ctx/u0/',
            256,
            num_filter_3x3=nf_3x3[-1],
            use_global_stats=use_global_stats,
            get_syms=False)

    # we won't use the first group
    groups = groups[1:]

    # build context layers
    upscales = [[4, 2], [2]]
    nf_proj = 32
    nf_upsamples = [[64, 64], [64]]
    ctx_layers = []
    for i, g in enumerate([group_ctx, groups[1]]):
        cl = []
        for j, (s, u) in enumerate(zip(upscales[i], nf_upsamples[i])):
            c = upsample_feature(
                g,
                name='ctx{}/{}/'.format(i, j),
                scale=s,
                num_filter_proj=nf_proj,
                num_filter_upsample=u,
                use_global_stats=use_global_stats)
            cl.append(c)
        ctx_layers.append(cl)
    ctx_layers = ctx_layers[::-1]

    # build multi scale feature layers
    from_layers = []
    nf_hyper = 384
    nf_hyper_proj = 128
    # small scale: hyperfeature
    hyper_names = ['hyper024/', 'hyper048/']
    nf_base = [nf_hyper - np.sum(np.array(i)) for i in nf_upsamples]
    for i, g in enumerate(groups[:2]):
        # gather all the upper layers
        g = relu_conv_bn(g, prefix_name='hyperproj/{}/'.format(i),
            num_filter=nf_base[i], kernel=(1, 1), pad=(0, 0),
            use_global_stats=use_global_stats)
        ctxi = [g]
        for j, c in enumerate(ctx_layers[i:]):
            ctxi.append(c[i])
        concat = mx.sym.concat(*(ctxi))
        hyper = relu_conv_bn(concat, prefix_name=hyper_names[i]+'1x1/',
            num_filter=nf_hyper_proj, kernel=(1, 1), pad=(0, 0),
            use_global_stats=use_global_stats)
        hyper = relu_conv_bn(hyper, prefix_name=hyper_names[i]+'3x3/',
            num_filter=nf_hyper, kernel=(3, 3), pad=(1, 1),
            use_global_stats=use_global_stats)
        from_layers.append(hyper)

    # remaining layers, bigger than 48
    for i, g in enumerate(groups[2:]):
        k = i + 2
        rf = int((2.0**k) * 24.0)
        prefix_name = 'hyper{}/'.format(rf)
        projh = relu_conv_bn(g, prefix_name='hyper{}/1x1/'.format(rf),
            num_filter=nf_hyper_proj, kernel=(1, 1), pad=(0, 0),
            use_global_stats=use_global_stats)
        convh = relu_conv_bn(projh, prefix_name='hyper{}/3x3/'.format(rf),
            num_filter=nf_hyper, kernel=(3, 3), pad=(1, 1),
            use_global_stats=use_global_stats)
        from_layers.append(convh)

    n_from_layers = len(from_layers)
    strides = []
    sizes = []
    sz_ratio = np.power(2.0, 1.0 / 4.0)
    for i in range(n_from_layers):
        st = 2 ** (i + 3)
        sz = st * 3.0
        strides.append(st)
        sizes.append([sz * sz_ratio, sz / sz_ratio])
    ratios = [[1.0, 1.0/2.0, 2.0, 1.0/3.0, 3.0]] * len(sizes)
    clip = False

    preds, anchors = multibox_layer_python(from_layers, n_classes,
            sizes=sizes, ratios=ratios, strides=strides, per_cls_reg=per_cls_reg, clip=False)
    return preds, anchors
