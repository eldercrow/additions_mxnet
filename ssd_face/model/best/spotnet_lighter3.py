import mxnet as mx
import numpy as np
from net_block_spotnet import *
from layer.multibox_prior_layer import MultiBoxPriorPython, MultiBoxPriorPythonProp


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
    bn_ = data
    nch = n_curr_ch
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


def clone_inception_group(data, prefix_group_name, src_syms):
    """
    inception unit, only full padding is supported
    """
    prefix_name = prefix_group_name
    incep_layers = []
    bn_ = data
    for ii in range(3):
        postfix_name = '3x3/{}'.format(ii + 1)
        bn_ = clone_relu_conv_bn(bn_, prefix_name + '3x3/{}/'.format(ii),
                                   src_syms['unit{}'.format(ii)])
        incep_layers.append(bn_)

    concat_ = mx.sym.concat(*incep_layers)
    concat_ = clone_relu_conv_bn(concat_, prefix_name + 'concat/', src_syms['concat'])

    if 'proj_data' in src_syms:
        # data = clone_conv(data, name=prefix_name+'proj/conv', src_layer=src_syms['proj_data'])
        data = clone_relu_conv_bn(
            data,
            prefix_name=prefix_name + 'proj/',
            src_syms=src_syms['proj_data'])
    return concat_ + data


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


def multibox_layer(from_layers,
                   num_classes,
                   sizes,
                   ratios,
                   strides,
                   use_global_stats,
                   clip=False,
                   clone_idx=[]):
    ''' multibox layer '''
    # parameter check
    assert len(from_layers) > 0, "from_layers must not be empty list"
    assert num_classes > 1, "num_classes {} must be larger than 1".format(
        num_classes)
    assert len(ratios) == len(from_layers), \
            "ratios and from_layers must have same length"
    assert len(sizes) == len(from_layers), \
            "sizes and from_layers must have same length"

    loc_pred_layers = []
    cls_pred_layers = []
    pred_layers = []
    anchor_layers = []
    #
    if len(clone_idx) > 1:
        clone_ref = clone_idx[0]
        clone_idx = clone_idx[1:]
    else:
        clone_ref = -1
        clone_idx = []

    for k, from_layer in enumerate(from_layers):
        from_name = from_layer.name
        num_anchors = len(sizes[k]) * len(ratios[k])
        num_loc_pred = num_anchors * 4
        num_cls_pred = num_anchors * num_classes

        if k == clone_ref:
            # from_layer, ref_fc = relu_conv_bn(
            #     from_layer,
            #     prefix_name='{}_fc/clone/'.format(from_name),
            #     num_filter=128,
            #     kernel=(1, 1),
            #     pad=(0, 0),
            #     use_global_stats=use_global_stats,
            #     get_syms=True)
            #
            from_layer = mx.sym.Activation(from_layer, act_type='relu')

            pred_conv = mx.sym.Convolution(
                from_layer,
                name='{}_pred/clone/conv'.format(from_name),
                num_filter=num_cls_pred+num_loc_pred,
                kernel=(3, 3),
                pad=(1, 1),
                no_bias=False)
            ref_pred = pred_conv
        elif k in clone_idx:
            # from_layer = clone_relu_conv_bn(
            #     from_layer,
            #     prefix_name='{}_fc/clone/'.format(from_name),
            #     src_syms=ref_fc)
            #
            from_layer = mx.sym.Activation(from_layer, act_type='relu')

            pred_conv = clone_conv(
                from_layer,
                name='{}_pred/clone/conv'.format(from_name),
                src_layer=ref_pred)
        else:
            # from_layer = relu_conv_bn(
            #     from_layer,
            #     prefix_name='{}_fc/'.format(from_name),
            #     num_filter=128,
            #     kernel=(1, 1),
            #     pad=(0, 0),
            #     use_global_stats=use_global_stats)
            #
            from_layer = mx.sym.Activation(from_layer, act_type='relu')

            pred_conv = mx.sym.Convolution(
                from_layer,
                name='{}_pred/conv'.format(from_name),
                num_filter=num_cls_pred+num_loc_pred,
                kernel=(3, 3),
                pad=(1, 1),
                no_bias=False)

        pred_conv = mx.sym.transpose(
            pred_conv, axes=(0, 2, 3, 1))  # (n h w ac), a=num_anchors
        pred_conv = mx.sym.reshape(
            pred_conv, shape=(0, -3, -4, num_anchors, -1))  # (n h*w a c)
        pred_conv = mx.sym.reshape(pred_conv, shape=(0, -3, -1))  # (n h*w*a c)
        pred_layers.append(pred_conv)

    anchors = mx.sym.Custom(
        *from_layers,
        op_type='multibox_prior_python',
        sizes=sizes,
        ratios=ratios,
        strides=strides,
        clip=int(clip))
    preds = mx.sym.concat(*pred_layers, num_args=len(pred_layers), dim=1)
    return [preds, anchors]


def get_spotnet(n_classes, patch_size, use_global_stats, n_group=5):
    """ main shared conv layers """
    data = mx.sym.Variable(name='data')

    conv1 = mx.sym.Convolution(
        data / 128.0,
        name='1/conv',
        num_filter=16,
        kernel=(3, 3),
        pad=(1, 1),
        no_bias=True)  # 32, 198
    concat1 = mx.sym.concat(conv1, -conv1, name='concat1')
    bn1 = mx.sym.BatchNorm(concat1, name='1/bn', use_global_stats=use_global_stats, fix_gamma=False)
    pool1 = pool(bn1)
    bn2 = relu_conv_bn(
        pool1,
        prefix_name='2/',
        num_filter=32,
        kernel=(3, 3),
        pad=(1, 1),
        use_crelu=True,
        use_global_stats=use_global_stats)

    nf_3x3 = ((32, 16, 16), (48, 24, 24), (64, 32, 32))  # nch: 128 192 256
    n_incep = (1, 1, 1)

    # basic groups
    group_i = bn2
    groups = []
    n_curr_ch = 64
    for i in range(len(nf_3x3)):
        group_i = pool(group_i)
        for j in range(n_incep[i]):
            group_i, n_curr_ch = inception_group(
                group_i,
                'g{}/u{}/'.format(i, j),
                n_curr_ch,
                num_filter_3x3=nf_3x3[i],
                use_crelu=True,
                use_global_stats=use_global_stats,
                get_syms=False)
        groups.append(group_i)

    pool4 = pool(groups[-1])

    # context layer
    nf_3x3_ctx = (64, 32, 32)
    group_ctx = pool4
    for i in range(2):
        group_ctx, n_curr_ch = inception_group(
            group_ctx,
            'g_ctx/u{}/'.format(i),
            n_curr_ch,
            num_filter_3x3=nf_3x3_ctx,
            use_global_stats=use_global_stats,
            get_syms=False)

    # build context layers
    upscales = [[8, 4, 2], [4, 2], [2]]
    nf_proj = 64
    nf_upsamples = [[32, 32, 32], [32, 32], [32]]
    ctx_layers = []
    for i, g in enumerate([group_ctx, groups[2], groups[1]]):
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

    # buffer layer for constructing clone layers
    clone_buffer = relu_conv_bn(
        pool4,
        prefix_name='clone_buffer/',
        num_filter=64,
        kernel=(1, 1),
        pad=(0, 0),
        use_global_stats=use_global_stats)

    # clone reference layer
    group_i = mx.sym.Activation(clone_buffer, act_type='relu')
    group_i = mx.sym.Convolution(group_i, name='g{}/proj/clone/conv'.format(len(groups)), 
            num_filter=64, kernel=(1,1), pad=(0,0), no_bias=True)
    sym_proj = group_i
    group_i, sym_dn = data_norm(group_i, name='dn/3', nch=64, get_syms=True)
    # group_i = mx.sym.InstanceNorm(group_i, name='dn/3')
    sym_in = group_i
    n_curr_ch = 64
    nf_3x3_ref = (32, 16, 16)
    syms_unit = []
    for j in range(1):
        group_i, n_curr_ch, syms = inception_group(
            group_i,
            'g{}/u{}/clone/'.format(len(groups), j),
            n_curr_ch,
            num_filter_3x3=nf_3x3_ref,
            use_global_stats=use_global_stats,
            get_syms=True)
        syms_unit.append(syms)
    groups.append(group_i)  # 96

    # cloned layers
    for i in range(len(groups), n_group):
        group_cloned = pool(groups[-1])
        group_cloned = mx.sym.Activation(group_cloned, act_type='relu')
        group_cloned = clone_conv(group_cloned, name='g{}/proj/clone/conv'.format(i), src_layer=sym_proj)
        group_cloned = data_norm(group_cloned, name='dn/{}'.format(i), nch=64, bias=sym_dn['bias'])
        # group_cloned = clone_in(group_cloned, name='dn/{}'.format(i), src_layer=sym_in)
        for j in range(1):
            group_cloned = clone_inception_group(
                group_cloned, 'g{}/u{}/clone/'.format(i, j), syms_unit[j])
        groups.append(group_cloned)  # 192 384 768

    from_layers = []
    nf_hyper = 192
    # small scale: hyperfeature
    hyper_names = ['hyper012/', 'hyper024/', 'hyper048/']
    nf_base = [nf_hyper - np.sum(np.array(i)) for i in nf_upsamples]
    for i, g in enumerate(groups[:3]):
        # gather all the upper layers
        g = relu_conv_bn(
            g,
            prefix_name='hyperproj/{}/'.format(i),
            num_filter=nf_base[i],
            kernel=(1, 1),
            pad=(0, 0),
            use_global_stats=use_global_stats)
        ctxi = [g]
        for j, c in enumerate(ctx_layers[i:]):
            ctxi.append(c[i])
        concat = mx.sym.concat(*(ctxi))
        hyper = relu_conv_bn(
            concat,
            prefix_name=hyper_names[i],
            num_filter=nf_hyper,
            kernel=(1, 1),
            pad=(0, 0),
            use_global_stats=use_global_stats)
        from_layers.append(hyper)

    # clone reference layer
    clone_ref = 3
    # groups[clone_ref], sym_dn_hyper = data_norm(groups[clone_ref], name='dn/hyper/3', nch=128, get_syms=True)
    # groups[clone_ref] = mx.sym.InstanceNorm(groups[clone_ref], name='dn/hyper/3')
    sym_in_hyper = groups[clone_ref]
    conv096, src_syms = relu_conv_bn(
        groups[clone_ref],
        prefix_name='hyper096/clone/',
        num_filter=nf_hyper,
        kernel=(1, 1),
        pad=(0, 0),
        use_global_stats=use_global_stats,
        get_syms=True)
    from_layers.append(conv096)

    # remaining clone layers
    clone_idx = [clone_ref]
    for i in range(clone_ref + 1, len(groups)):
        rf = int((2.0**i) * 12.0)
        prefix_name = 'hyper{}/clone/'.format(rf)
        # groups[i] = data_norm(groups[i], name='dn/hyper/{}'.format(i), nch=128, bias=sym_dn_hyper['bias'])
        # groups[i] = clone_in(groups[i], name='dn/hyper/{}'.format(i), src_layer=sym_in_hyper)
        conv_ = clone_relu_conv_bn(
            groups[i], prefix_name=prefix_name, src_syms=src_syms)
        from_layers.append(conv_)
        clone_idx.append(i)

    n_from_layers = len(from_layers)
    strides = [2**(i + 2) for i in range(n_from_layers)]
    sizes = []
    sz_ratio = np.power(2.0, 1.0 / 4.0)
    for i in range(n_from_layers):
        s = 12.0 * (2.0**i)
        sizes.append([s * sz_ratio, s / sz_ratio])
    ratios = [[1.0,]] * len(sizes)
    clip = False

    preds, anchors = multibox_layer(
        from_layers,
        n_classes,
        sizes=sizes,
        ratios=ratios,
        strides=strides,
        use_global_stats=use_global_stats,
        clip=clip,
        clone_idx=clone_idx)
    return preds, anchors
