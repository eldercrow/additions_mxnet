import mxnet as mx
from net_block_clone import *
from multibox_prior_layer import *

def inception_group(data, prefix_group_name, n_curr_ch, 
        num_filter_3x3, use_crelu=False, use_dn=False, 
        use_global_stats=False, get_syms=False):
    """ 
    inception unit, only full padding is supported
    """
    # save symbols anyway
    syms = {}

    prefix_name = prefix_group_name + '/'
    incep_layers = []
    conv_ = data
    nch = n_curr_ch
    num_filter_incep = 0
    for ii in range(3):
        postfix_name = '3x3/' + str(ii+1)
        conv_, s = bn_relu_conv(conv_, prefix_name, postfix_name, 
                num_filter=num_filter_3x3[ii], kernel=(3,3), pad=(1,1), use_crelu=use_crelu, 
                use_dn=use_dn, nch=nch, 
                use_global_stats=use_global_stats, get_syms=True)
        syms['unit{}'.format(ii+1)] = s
        incep_layers.append(conv_)
        nch = num_filter_3x3[ii]
        if use_crelu:
            nch *= 2
        num_filter_incep += nch

    concat_ = mx.sym.concat(*incep_layers)

    if num_filter_incep != n_curr_ch:
        data, s = bn_relu_conv(data, prefix_name, 'proj/', 
                num_filter=num_filter_incep, kernel=(1,1), 
                use_dn=use_dn, nch=n_curr_ch, 
                use_global_stats=use_global_stats, get_syms=True)
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
    prefix_name = prefix_group_name + '/'
    incep_layers = []
    conv_ = data
    for ii in range(3):
        postfix_name = '3x3/{}'.format(ii+1)
        conv_ = clone_bn_relu_conv(conv_, prefix_name, postfix_name, src_syms['unit{}'.format(ii+1)])
        incep_layers.append(conv_)

    concat_ = mx.sym.concat(*incep_layers)

    if 'proj_data' in src_syms:
        data = clone_bn_relu_conv(data, prefix_name=prefix_name+'proj/', src_syms=src_syms['proj_data'])
    return concat_ + data

def build_hyperfeature(layer, ctx_layer, name, num_filter_hyper, use_global_stats):
    """
    """
    concat_ = mx.sym.concat(layer, ctx_layer)
    return bn_relu_conv(concat_, prefix_name=name+'/hyper/', 
            num_filter=num_filter_hyper, kernel=(1,1), pad=(0,0),
            use_global_stats=use_global_stats)

def upsample_feature(data, name, scale, num_filter_proj=0, num_filter_upsample=0, use_global_stats=False):
    ''' use subpixel_upsample to upsample a given layer '''
    if num_filter_proj > 0:
        proj = bn_relu_conv(data, prefix_name=name+'/proj/', 
                num_filter=num_filter_proj, kernel=(1,1), pad=(0,0), 
                use_global_stats=use_global_stats)
    else:
        proj = data
    nf = num_filter_upsample * scale * scale
    conv = bn_relu_conv(proj, prefix_name=name+'/conv/', 
            num_filter=nf, kernel=(3,3), pad=(1,1), 
            use_global_stats=use_global_stats)
    return subpixel_upsample(conv, num_filter_upsample, scale, scale)

def multibox_layer(from_layers, num_classes, sizes, ratios, strides, use_global_stats, clip=False, clone_idx=[]):
    ''' multibox layer '''
    # parameter check
    assert len(from_layers) > 0, "from_layers must not be empty list"
    assert num_classes > 1, "num_classes {} must be larger than 1".format(num_classes)
    assert len(ratios) == len(from_layers), "ratios and from_layers must have same length"
    assert len(sizes) == len(from_layers), "sizes and from_layers must have same length"

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
            pred_conv, ref_syms = bn_relu_conv(from_layer, prefix_name='{}_pred/'.format(from_name), 
                    num_filter=num_loc_pred+num_cls_pred, kernel=(3,3), pad=(1,1), no_bias=False, 
                    use_dn=True, nch=128, 
                    use_global_stats=use_global_stats, get_syms=True) # (n ac h w)
        elif k in clone_idx:
            pred_conv = clone_bn_relu_conv(from_layer, prefix_name='{}_pred/'.format(from_name), 
                    src_syms=ref_syms)
        else:
            pred_conv = bn_relu_conv(from_layer, prefix_name='{}_pred/'.format(from_name), 
                    num_filter=num_loc_pred+num_cls_pred, kernel=(3,3), pad=(1,1), no_bias=False, 
                    use_global_stats=use_global_stats) # (n ac h w)

        pred_conv = mx.sym.transpose(pred_conv, axes=(0, 2, 3, 1)) # (n h w ac), a=num_anchors
        pred_conv = mx.sym.reshape(pred_conv, shape=(0, -3, -4, num_anchors, -1)) # (n h*w a c)
        pred_conv = mx.sym.reshape(pred_conv, shape=(0, -3, -1)) # (n h*w*a c)
        pred_layers.append(pred_conv)

    anchors = mx.sym.Custom(*from_layers, op_type='multibox_prior_python', 
            sizes=sizes, ratios=ratios, strides=strides, clip=int(clip))
    preds = mx.sym.concat(*pred_layers, num_args=len(pred_layers), dim=1)
    return [preds, anchors]

def get_phgnet(n_classes, patch_size, use_global_stats, n_group=5):
    """ main shared conv layers """
    data = mx.sym.Variable(name='data')

    conv1_1 = mx.sym.Convolution(data/128.0, name='conv1/1', num_filter=16, 
            kernel=(3,3), pad=(1,1), no_bias=True) # 32, 198
    concat1_1 = mx.sym.concat(conv1_1, -conv1_1, name='concat1/1')
    conv1_2 = bn_relu_conv(concat1_1, postfix_name='1/2', 
            num_filter=32, kernel=(3,3), pad=(1,1), use_crelu=True, 
            use_global_stats=use_global_stats) # 48, 196 
    conv1_3 = bn_relu_conv(conv1_2, postfix_name='1/3', 
            num_filter=32, kernel=(3,3), pad=(1,1), use_crelu=True, 
            use_global_stats=use_global_stats) # 48, 192 

    nf_3x3 = ((16, 8, 8), (24, 12, 12), (48, 24, 24)) # nch: 64 96 96
    n_incep = (2, 2, 2)

    # basic groups
    group_i = conv1_3
    groups = []
    n_curr_ch = 64
    for i in range(len(nf_3x3)):
        group_i = pool(group_i, kernel=(2,2), name='pool{}'.format(i+1)) # 48 24 12 6
        for j in range(n_incep[i]):
            group_i, n_curr_ch = inception_group(group_i, 'g{}/u{}'.format(i+1, j+1), n_curr_ch, 
                    num_filter_3x3=nf_3x3[i], use_crelu=(i < 2), 
                    use_global_stats=use_global_stats, get_syms=False) 
        # experimental, 6px
        if i == 0:
            group_0 = bn_relu_conv(conv1_3, prefix_name='g0/', 
                    num_filter=64, kernel=(2,2), pad=(0,0), stride=(2,2), 
                    use_global_stats=use_global_stats)
            groups.append(group_0 + pool(conv1_3))
        groups.append(group_i)

    # context layer
    n_curr_ch = 96
    nf_3x3_ctx = (64, 32, 32)
    group_ctx = pool(groups[-1])
    for i in range(2):
        group_ctx, n_curr_ch = inception_group(group_ctx, 'g_ctx/u{}'.format(i+1), n_curr_ch,
                num_filter_3x3=nf_3x3_ctx, use_crelu=False, use_dn=False, 
                use_global_stats=use_global_stats, get_syms=False)

    # layers for hourglass model
    ctx_scales = (1, 2, 4, 8)
    ctx_layers = []
    for i, g in enumerate((groups[1], groups[2], groups[3], group_ctx)):
        ctx_layer = upsample_feature(g, name='ctx{}'.format(i+1), scale=ctx_scales[i], 
                num_filter_proj=32, num_filter_upsample=8, use_global_stats=use_global_stats)
        ctx_layers.append(ctx_layer)

    # buffer layer for constructing clone layers
    clone_buffer = bn_relu_conv(groups[-1], prefix_name='clone_buffer/', 
            num_filter=128, kernel=(1,1), pad=(0,0), 
            use_global_stats=use_global_stats)

    # clone reference layer
    nf_3x3_ref = (64, 32, 32)
    group_i, syms_proj = bn_relu_conv(clone_buffer, prefix_name='g{}/proj/'.format(len(groups)),
            num_filter=64, kernel=(3,3), pad=(1,1),
            use_dn=True, nch=128, 
            use_global_stats=use_global_stats, get_syms=True)
    group_i = pool(group_i)
    n_curr_ch = 64
    syms_unit = []
    for j in range(2):
        group_i, n_curr_ch, syms = inception_group(group_i, 'g{}/u{}'.format(len(groups), j+1), n_curr_ch, 
                num_filter_3x3=nf_3x3_ref, use_dn=True, 
                use_global_stats=use_global_stats, get_syms=True) 
        syms_unit.append(syms)
    groups.append(group_i)

    # cloned layers
    group_cloned = groups[-1]
    for i in range(len(groups), n_group):
        group_cloned = clone_bn_relu_conv(group_cloned, 'g{}/proj'.format(i), '', syms_proj)
        group_cloned = pool(group_cloned)
        for j in range(2):
            group_cloned = clone_inception_group(group_cloned, 'g{}/u{}'.format(i, j+1),  
                    syms_unit[j])
        groups.append(group_cloned)

    from_layers = []
    # small scale: hyperfeature
    hyper_names = ['hyper006/', 'hyper012/', 'hyper024/', 'hyper048/']
    ctx_scales = (1, 1, 2, 4)
    for i, g in enumerate(groups[:4]):
        # gather all the upper layers
        ctxi = []
        s = ctx_scales[i]
        for j, c in enumerate(ctx_layers[i:]):
            if s > 1:
                ctxi.append(pool(c, kernel=(s, s), stride=(s, s), pool_type='avg'))
            else:
                ctxi.append(c)
        concat = mx.sym.concat(*([g] + ctxi))
        hyper = bn_relu_conv(concat, prefix_name=hyper_names[i], num_filter=128, 
                kernel=(1,1), pad=(0,0), use_global_stats=use_global_stats)
        from_layers.append(hyper)

    # clone reference layer
    clone_ref = 4
    conv096, src_syms = bn_relu_conv(groups[clone_ref], prefix_name='hyper096/conv/', 
            num_filter=128, kernel=(1,1), pad=(0,0), 
            use_dn=True, nch=128, 
            use_global_stats=use_global_stats, get_syms=True)
    from_layers.append(conv096)

    # remaining clone layers
    clone_idx = [clone_ref]
    for i in range(clone_ref+1, len(groups)):
        rf = int((2.0**i) * 6.0)
        prefix_name = 'hyper{}/conv/'.format(rf)
        conv_ = clone_bn_relu_conv(groups[i], prefix_name=prefix_name, src_syms=src_syms)
        from_layers.append(conv_)
        clone_idx.append(i)

    n_from_layers = len(from_layers)
    strides = [2**i for i in range(n_from_layers)]
    strides[0] = 2
    sizes = []
    sz_ratio = np.power(2.0, 1.0 / 3.0)
    for i in range(n_from_layers):
        s = 6.0 * (2.0**i)
        sizes.append([s, s*sz_ratio, s/sz_ratio])
    ratios = [[1.0,]] * len(sizes)
    clip = False

    preds, anchors = multibox_layer(from_layers, n_classes, 
            sizes=sizes, ratios=ratios, strides=strides, 
            use_global_stats=use_global_stats, clip=clip, clone_idx=clone_idx)
    return preds, anchors