import mxnet as mx
from net_block_mb import *
from multibox_prior_layer import *


def inception_group(data,
                    prefix_group_name,
                    nf3,
                    nf1,
                    strides,
                    num_group=1,
                    use_global_stats=False,
                    get_syms=False):
    """
    inception unit, only full padding is supported
    """
    # save symbols anyway
    syms = {}

    prefix_name = prefix_group_name
    conv_ = data
    for ii in range(3):
        conv_, s = conv_sep_unit(
            conv_,
            prefix_name=prefix_name + '{}/'.format(ii),
            nf3=nf3, 
            nf1=nf1,
            stride=strides[ii],
            num_group=num_group,
            use_global_stats=use_global_stats,
            get_syms=True)
        syms['unit{}'.format(ii)] = s

    if strides[0][0] != 1:
        data = syms['unit0']['conv']
        syms['proj_data'] = data

    res_ = conv_ + data

    if get_syms:
        return res_, nf1, syms
    else:
        return res_, nf1


def clone_inception_group(data, prefix_group_name, src_syms):
    """
    inception unit, only full padding is supported
    """
    prefix_name = prefix_group_name
    conv_ = data
    for ii in range(3):
        conv_ = clone_conv_sep_unit(conv_, prefix_name + '{}/'.format(ii),
                                    src_syms['unit{}'.format(ii)])
        # for projection
        if ii == 0 and 'proj_data' in src_syms:
            data = conv_

    return conv_ + data


# def build_hyperfeature(layer, ctx_layer, name, num_filter_ctx,
#                        num_filter_hyper, use_global_stats):
#     """
#     """
#     ctx_ = upsample_feature(
#         ctx_layer,
#         name=name + '/upconv',
#         scale=2,
#         num_filter_proj=64,
#         num_filter_upsample=num_filter_ctx,
#         use_global_stats=use_global_stats)
#     layer_ = bn_relu_conv(
#         layer,
#         prefix_name=name + '/proj/',
#         num_filter=num_filter_hyper - num_filter_ctx,
#         kernel=(1, 1),
#         pad=(0, 0),
#         use_global_stats=use_global_stats)
#     concat_ = mx.sym.concat(layer_, ctx_)
#     return concat_


def upsample_feature(data,
                     name,
                     scale,
                     num_filter_proj=0,
                     num_filter_upsample=0,
                     use_global_stats=False):
    ''' use subpixel_upsample to upsample a given layer '''
    if num_filter_proj > 0:
        proj = bn_relu_conv(
            data,
            prefix_name=name + 'proj/',
            num_filter=num_filter_proj,
            kernel=(1, 1),
            pad=(0, 0),
            use_global_stats=use_global_stats)
    else:
        proj = data
    nf = num_filter_upsample * scale * scale
    conv = conv_sep_unit(
        proj,
        prefix_name=name + 'conv/',
        nf3=num_filter_proj,
        nf1=nf,
        num_group=4,
        use_global_stats=use_global_stats)
    # conv = bn_relu_conv(
    #     proj,
    #     prefix_name=name + 'conv/',
    #     num_filter=nf,
    #     kernel=(3, 3),
    #     pad=(1, 1),
    #     use_global_stats=use_global_stats)
    return subpixel_upsample(conv, num_filter_upsample, scale, scale)


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
    assert len(ratios) == len(
        from_layers), "ratios and from_layers must have same length"
    assert len(sizes) == len(
        from_layers), "sizes and from_layers must have same length"

    loc_pred_layers = []
    cls_pred_layers = []
    pred_layers = []
    anchor_layers = []
    # num_classes += 1 # always use background as label 0
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
            # fc_as_conv, ref_fc1 = bn_relu_conv(
            #     from_layer,
            #     prefix_name='{}_fc1/'.format(from_name),
            #     num_filter=64,
            #     kernel=(3, 3),
            #     pad=(1, 1),
            #     use_global_stats=use_global_stats,
            #     get_syms=True)
            fc_as_conv, ref_fc2 = bn_relu_conv(
                from_layer,
                prefix_name='{}_fc/clone/'.format(from_name),
                num_filter=256,
                kernel=(1, 1),
                pad=(0, 0),
                use_global_stats=use_global_stats,
                get_syms=True)
            pred_conv, ref_syms = bn_relu_conv(
                fc_as_conv,
                prefix_name='{}_pred/clone/'.format(from_name),
                num_filter=num_loc_pred + num_cls_pred,
                kernel=(3, 3),
                pad=(1, 1),
                no_bias=False,
                use_global_stats=use_global_stats,
                get_syms=True)  # (n ac h w)
        elif k in clone_idx:
            # fc_as_conv = clone_bn_relu_conv(
            #     from_layer,
            #     prefix_name='{}_fc1/'.format(from_name),
            #     src_syms=ref_fc1)
            fc_as_conv = clone_bn_relu_conv(
                from_layer,
                prefix_name='{}_fc/clone/'.format(from_name),
                src_syms=ref_fc2)
            pred_conv = clone_bn_relu_conv(
                fc_as_conv,
                prefix_name='{}_pred/clone/'.format(from_name),
                src_syms=ref_syms)
        else:
            # fc_as_conv = bn_relu_conv(
            #     from_layer,
            #     prefix_name='{}_fc1/'.format(from_name),
            #     num_filter=64,
            #     kernel=(3, 3),
            #     pad=(1, 1),
            #     use_global_stats=use_global_stats)
            fc_as_conv = bn_relu_conv(
                from_layer,
                prefix_name='{}_fc/'.format(from_name),
                num_filter=256,
                kernel=(1, 1),
                pad=(0, 0),
                use_global_stats=use_global_stats)
            # fc_as_conv = mx.sym.Dropout(fc_as_conv, p=0.25)
            pred_conv = bn_relu_conv(
                fc_as_conv,
                prefix_name='{}_pred/'.format(from_name),
                num_filter=num_loc_pred + num_cls_pred,
                kernel=(3, 3),
                pad=(1, 1),
                no_bias=False,
                use_global_stats=use_global_stats)  # (n ac h w)

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


def get_spotnet_sep(n_classes, patch_size, use_global_stats, n_group=5):
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
    # pool1 = pool(concat1)
    conv2 = bn_relu_conv(
        concat1,
        prefix_name='2_0/',
        num_filter=32,
        kernel=(3, 3),
        pad=(1, 1),
        stride=(2, 2),
        use_crelu=True,
        use_global_stats=use_global_stats)
    conv2 = bn_relu_conv(
        conv2,
        prefix_name='2_1/',
        num_filter=64,
        kernel=(1, 1),
        pad=(0, 0),
        use_global_stats=use_global_stats)

    num_filters = (128, 192, 256)
    n_incep = (2, 2, 2)
    strides = (((2, 2), (1, 1), (1, 1)), ((1, 1), (1, 1), (1, 1)))
    num_groups = (2, 2, 4)

    # basic groups
    # groups = [pool(subsample_pool(conv1_3, name='pool0', num_filter=64, use_global_stats=use_global_stats))]
    # sub1_3 = subsample_pool(conv1_3, num_filter=16, name='sub0', use_global_stats=use_global_stats)
    # pool1_3 = convaspool(sub1_3, num_filter=64, name='pool0', use_global_stats=use_global_stats)
    # groups = [pool1_3]  # 384
    group_i = conv2
    groups = []
    n_curr_ch = 64
    for i in range(len(num_filters)):
        for j in range(n_incep[i]):
            group_i, n_curr_ch = inception_group(
                group_i,
                'g{}/u{}/'.format(i, j),
                nf3=num_filters[i] / 2,
                nf1=num_filters[i],
                strides=strides[j],
                num_group=num_groups[i],
                use_global_stats=use_global_stats,
                get_syms=False)
        groups.append(group_i)

    # context layer
    n_curr_ch = 256
    nf_ctx = 128
    group_ctx = groups[-1]
    for i in range(2):
        group_ctx, n_curr_ch = inception_group(
            group_ctx,
            'g_ctx/u{}/'.format(i),
            nf3=n_curr_ch / 2,
            nf1=nf_ctx,
            strides=strides[i],
            num_group=4,
            use_global_stats=use_global_stats,
            get_syms=False)
    # group_ctx = pool(group_ctx)

    # build context layers
    upscales = [[8, 4, 2], [4, 2], [2]]
    nf_proj = 128
    nf_upsample = 32
    ctx_layers = []
    for i, g in enumerate([group_ctx, groups[2], groups[1]]):
        cl = []
        for j, s in enumerate(upscales[i]):
            c = upsample_feature(
                g,
                name='ctx{}/{}/'.format(i, j),
                scale=s,
                num_filter_proj=nf_proj,
                num_filter_upsample=nf_upsample,
                use_global_stats=use_global_stats)
            cl.append(c)
        ctx_layers.append(cl)
    ctx_layers = ctx_layers[::-1]

    # buffer layer for constructing clone layers
    clone_buffer = bn_relu_conv(
        groups[-1],
        prefix_name='clone_buffer/',
        num_filter=64,
        kernel=(1, 1),
        pad=(0, 0),
        use_global_stats=use_global_stats)

    # clone reference layer
    group_i, syms_proj = bn_relu_conv(
        clone_buffer,
        prefix_name='g{}/proj/clone/'.format(len(groups)),
        num_filter=64,
        kernel=(1, 1),
        pad=(0, 0),
        use_global_stats=use_global_stats,
        get_syms=True)
    n_curr_ch = 64
    nf_ref = 64
    syms_unit = []
    for j in range(2):
        group_i, n_curr_ch, syms = inception_group(
            group_i,
            'g{}/u{}/clone/'.format(len(groups), j),
            nf3=n_curr_ch / 2,
            nf1=nf_ref,
            strides=strides[j],
            num_group=4,
            use_global_stats=use_global_stats,
            get_syms=True)
        syms_unit.append(syms)
    groups.append(group_i)  # 96

    # cloned layers
    for i in range(len(groups), n_group):
        group_cloned = groups[-1]
        group_cloned = clone_bn_relu_conv(
            group_cloned, 'g{}/proj/clone/'.format(i), syms_proj)
        for j in range(2):
            group_cloned = clone_inception_group(
                group_cloned, 'g{}/u{}/clone/'.format(i, j), syms_unit[j])
        groups.append(group_cloned)  # 192 384 768

    from_layers = []
    nf_hyper = 128
    # small scale: hyperfeature
    hyper_names = ['hyper012/', 'hyper024/', 'hyper048/']
    nf_base = [nf_hyper - i * nf_upsample for i in range(3, 0, -1)]
    for i, g in enumerate(groups[:3]):
        # gather all the upper layers
        g = bn_relu_conv(
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
        hyper = bn_relu_conv(
            concat,
            prefix_name=hyper_names[i],
            num_filter=nf_hyper,
            kernel=(1, 1),
            pad=(0, 0),
            use_global_stats=use_global_stats)
        from_layers.append(hyper)
    # # small scale: hyperfeature
    # hyper = build_hyperfeature(groups[2], group_ctx, name='hyper048',
    #         num_filter_ctx=48, num_filter_hyper=128, use_global_stats=use_global_stats)
    # from_layers.insert(0, hyper)
    # hyper_names = ['hyper024', 'hyper012']
    # nf_ctx = [96, 192]
    # for i, g in enumerate([groups[1], groups[0]]):
    #     hyper = build_hyperfeature(g, from_layers[0], name=hyper_names[i],
    #             num_filter_ctx=nf_ctx[i], num_filter_hyper=256,
    #             use_global_stats=use_global_stats)
    #     from_layers.insert(0, hyper)

    # clone reference layer
    clone_ref = 3
    conv096, src_syms = bn_relu_conv(
        groups[clone_ref],
        prefix_name='hyper096/',
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
        prefix_name = 'hyper{}/'.format(rf)
        conv_ = clone_bn_relu_conv(
            groups[i], prefix_name=prefix_name, src_syms=src_syms)
        from_layers.append(conv_)
        clone_idx.append(i)

    n_from_layers = len(from_layers)
    strides = [2**(i + 2) for i in range(n_from_layers)]
    sizes = []
    sz_ratio = np.power(2.0, 1.0 / 3.0)
    for i in range(n_from_layers):
        s = 12.0 * (2.0**i)
        sizes.append([s, s * sz_ratio, s / sz_ratio])
    ratios = [[
        1.0,
    ]] * len(sizes)
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
