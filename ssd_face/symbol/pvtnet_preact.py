import mxnet as mx
from net_block_clone import *

def inception_group(data, prefix_group_name, n_curr_ch,
        num_filter_3x3, num_filter_1x1, use_crelu=False, 
        use_global_stats=False, fix_gamma=True, get_syms=False):
    """ 
    inception unit, only full padding is supported
    """
    syms = {}
    prefix_name = prefix_group_name + '/'

    incep_layers = []
    conv_ = data
    for ii in range(2):
        postfix_name = '3x3/' + str(ii+1)
        conv_, s = bn_relu_conv(conv_, prefix_name, postfix_name, 
                num_filter=num_filter_3x3, kernel=(3,3), pad=(1,1), use_crelu=use_crelu, 
                use_global_stats=use_global_stats, fix_gamma=fix_gamma, get_syms=True)
        syms['unit{}'.format(ii)] = s
        incep_layers.append(conv_)
    # poolup2 layer
    postfix_name = '3x3/3'
    conv_, s = bn_relu_conv_poolup2(conv_, prefix_name, postfix_name, 
            num_filter=num_filter_3x3, kernel=(3,3), pad=(1,1), use_crelu=use_crelu, 
            use_global_stats=use_global_stats, fix_gamma=fix_gamma, get_syms=True)
    syms['unit3'] = s
    incep_layers.append(conv_)

    res_ = mx.sym.concat(*incep_layers)

    if get_syms:
        return res_, num_filter_1x1, syms
    else:
        return res_, num_filter_1x1

def clone_inception_group(data, prefix_group_name, src_syms): 
    """ 
    inception unit, only full padding is supported
    """
    prefix_name = prefix_group_name + '/'

    incep_layers = []
    conv_ = data
    for ii in range(2):
        postfix_name = '3x3/' + str(ii+1)
        conv_ = clone_bn_relu_conv(conv_, prefix_name, postfix_name, src_syms=src_syms['unit{}'.format(ii)])
        incep_layers.append(conv_)

    postfix_name = '3x3/3'
    conv_ = clone_bn_relu_conv_poolup2(conv_, prefix_name, src_syms=src_syms['unit3'])
    incep_layers.append(conv_)

    res_ = mx.sym.concat(*incep_layers)
    # concat_ = mx.sym.concat(*incep_layers)
    # res_ = clone_bn_relu_conv(concat_, prefix_name, '1x1', src_syms=src_syms['proj'])
    return res_

def get_pvtnet_preact(use_global_stats, fix_gamma=False, n_group=5):
    """ main shared conv layers """
    data = mx.sym.Variable(name='data')

    conv1_1 = mx.sym.Convolution(data / 128.0, name='conv1/1', 
            num_filter=16, kernel=(3,3), pad=(1,1), no_bias=True) 
    conv1_2 = bn_relu_conv(conv1_1, postfix_name='1/2', 
            num_filter=32, kernel=(3,3), pad=(1,1), use_crelu=True, 
            use_global_stats=use_global_stats, fix_gamma=fix_gamma) 
    conv1_3 = bn_relu_conv(conv1_2, postfix_name='1/3', 
            num_filter=64, kernel=(3,3), pad=(1,1), use_crelu=True, 
            use_global_stats=use_global_stats, fix_gamma=fix_gamma) 
    concat1 = mx.sym.concat(conv1_2, conv1_3)

    nf_3x3 = [16, 32, 48, 64, 64] # 12 24 48 96 192
    nf_1x1 = [16*3, 32*3, 48*3, 64*3, 64*3]

    group_i = concat1
    groups = []
    n_curr_ch = 96
    for i in range(5):
        group_i = pool(group_i) 
        # syms will be overwritten but it's ok we'll use the last one anyway
        group_i, n_curr_ch, syms = inception_group(group_i, 'g{}/u1'.format(i+1), n_curr_ch, 
                num_filter_3x3=nf_3x3[i], num_filter_1x1=nf_1x1[i], use_crelu=(i == 0), 
                use_global_stats=use_global_stats, fix_gamma=fix_gamma, get_syms=True) 
        groups.append(group_i)

    # for context feature
    n_curr_ch = nf_1x1[-2]
    nf_3x3_ctx = 32
    nf_1x1_ctx = 32*3
    group_c = pool(groups[-2])
    group_c, n_curr_ch = inception_group(group_c, 'g_ctx/u1', n_curr_ch, 
            num_filter_3x3=nf_3x3_ctx, num_filter_1x1=nf_1x1_ctx,
            use_global_stats=use_global_stats, fix_gamma=fix_gamma)

    # upsample feature for small face (12px)
    conv0 = bn_relu_conv(groups[1], prefix_name='group0/', 
            num_filter=64, kernel=(3,3), pad=(1,1), 
            use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    convu = subpixel_upsample(conv0, 16, 2, 2)
    groups[0] = mx.sym.concat(groups[0], convu)

    # cloned feature
    for i in range(5, n_group):
        group_i = pool(group_i)
        group_i = clone_inception_group(group_i, 'g{}/u1'.format(i+1), syms)
        groups.append(group_i)

    return groups, group_c
