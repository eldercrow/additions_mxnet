# import find_mxnet
import mxnet as mx
from net_block_clone import *

def inception_group(data, prefix_group_name, n_curr_ch, 
        num_filter_3x3, use_crelu=False, use_dn=False, 
        use_global_stats=False, fix_gamma=True, get_syms=False):
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
                use_global_stats=use_global_stats, fix_gamma=fix_gamma, get_syms=True)
        syms['unit{}'.format(ii)] = s
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
                use_global_stats=use_global_stats, fix_gamma=fix_gamma, get_syms=True)
        syms['proj_data'] = s

    res_ = concat_ + data

    if get_syms:
        return res_, num_filter_incep, syms
    else:
        return res_, num_filter_incep

def clone_inception_group(data, prefix_group_name, src_syms):
    '''
    clone an inception group
    '''
    prefix_name = prefix_group_name + '/'
    incep_layers = []
    conv_ = data
    for ii in range(3):
        postfix_name = '3x3/{}'.format(ii+1)
        conv_ = clone_bn_relu_conv(conv_, prefix_name, postfix_name, src_syms['unit{}'.format(ii)])
        incep_layers.append(conv_)

    concat_ = mx.sym.concat(*incep_layers)

    if 'proj_data' in src_syms:
        data = clone_bn_relu_conv(data, prefix_name+'proj/', src_syms['proj_data'])
    return concat_ + data

def get_hjnet_preact(use_global_stats, fix_gamma=True, n_group=4):
    """ main shared conv layers """
    data = mx.sym.Variable(name='data')
    data_ = data / 128.0

    conv1_1 = mx.sym.Convolution(data_, name='conv1/1', num_filter=12, 
            kernel=(3,3), pad=(1,1), no_bias=True) # 32, 198
    concat1_1 = mx.sym.concat(conv1_1, -conv1_1, name='concat1/1')
    conv1_2 = bn_relu_conv(concat1_1, postfix_name='1/2', 
            num_filter=16, kernel=(3,3), pad=(1,1), use_crelu=True, 
            use_global_stats=use_global_stats, fix_gamma=fix_gamma) # 48, 196 
    conv1_3 = bn_relu_conv(conv1_2, postfix_name='1/3', 
            num_filter=24, kernel=(3,3), pad=(1,1), use_crelu=True, 
            use_global_stats=use_global_stats, fix_gamma=fix_gamma) # 48, 192 

    nf_3x3 = ((12, 6, 6), (16, 8, 8), (32, 16, 16), (32, 16, 16)) # 12 24 48 96
    n_incep = (2, 2, 2, 2)

    group_i = conv1_3
    groups = []
    syms_group = []
    n_curr_ch = 48
    for i in range(4):
        syms_unit = []
        group_i = pool(group_i, kernel=(2,2), name='pool{}'.format(i+1)) # 48 24 12 6
        for j in range(n_incep[i]):
            group_i, n_curr_ch, syms = inception_group(group_i, 'g{}/u{}'.format(i+1, j+1), n_curr_ch, 
                    num_filter_3x3=nf_3x3[i], use_crelu=(i < 2), use_dn=(i==3), 
                    use_global_stats=use_global_stats, fix_gamma=fix_gamma, get_syms=True) 
            syms_unit.append(syms)
        groups.append(group_i)
        syms_group.append(syms_unit)

    # for context feature
    n_curr_ch = 64
    nf_3x3_ctx = (32, 16, 16)
    group_c = pool(groups[2])
    for i in range(2):
        group_c, n_curr_ch, s = inception_group(group_c, 'g_ctx/u{}'.format(i+1), n_curr_ch,
                num_filter_3x3=nf_3x3_ctx, use_crelu=False, use_dn=False, 
                use_global_stats=use_global_stats, fix_gamma=fix_gamma, get_syms=True)

    # upsample feature for small face (12px)
    conv0 = bn_relu_conv(groups[1], prefix_name='group0/', 
            num_filter=48, kernel=(3,3), pad=(1,1), 
            use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    convu = subpixel_upsample(conv0, 12, 2, 2)
    groups[0] = mx.sym.concat(groups[0], convu)

    # cloned layers
    group_cloned = groups[-1]
    syms_unit = syms_group[-1]
    for i in range(4, n_group):
        group_cloned = pool(group_cloned, kernel=(2,2), name='pool{}'.format(i+1+len(n_incep)))
        for j in range(n_incep[-1]):
            group_cloned = clone_inception_group(group_cloned, 'g{}/u{}'.format(i+1+len(n_incep), j+1), 
                    syms_unit[j])
        groups.append(group_cloned)

    return groups, group_c
