# import find_mxnet
import mxnet as mx
from net_block_clone import *

def inception_group(data, prefix_group_name, n_curr_ch, 
        num_filter_3x3, num_filter_1x1, 
        use_global_stats=False, fix_gamma=True, get_syms=False):
    """ 
    inception unit, only full padding is supported
    """
    # save symbols anyway
    syms = {}

    prefix_name = prefix_group_name + '/'
    incep_layers = []
    conv_ = data
    for ii in range(2):
        postfix_name = '3x3/' + str(ii+1)
        conv_, s = bn_relu_conv(conv_, prefix_name, postfix_name, 
                num_filter=num_filter_3x3, kernel=(3,3), pad=(1,1), 
                use_global_stats=use_global_stats, fix_gamma=fix_gamma, get_syms=True)
        syms['unit{}'.format(ii)] = s
        incep_layers.append(conv_)
    # poolup2 layer
    postfix_name = '3x3/3'
    conv_, s = bn_relu_conv_poolup2(conv_, prefix_name, postfix_name, 
            num_filter=num_filter_3x3, kernel=(3,3), pad=(1,1), 
            use_global_stats=use_global_stats, fix_gamma=fix_gamma, get_syms=True)
    syms['unit3'] = s
    incep_layers.append(conv_)

    concat_ = mx.sym.concat(*incep_layers)

    if (num_filter_3x3 * 3) != num_filter_1x1:
        concat_, s = bn_relu_conv(concat_, prefix_name, '1x1', 
                num_filter=num_filter_1x1, kernel=(1,1), 
                use_global_stats=use_global_stats, fix_gamma=fix_gamma, get_syms=True)
        sums['proj_concat'] = s
    
    if n_curr_ch != num_filter_1x1:
        data, s = bn_relu_conv(data, prefix_name+'proj/', 
                num_filter=num_filter_1x1, kernel=(1,1), 
                use_global_stats=use_global_stats, fix_gamma=fix_gamma, get_syms=True)
        syms['proj_data'] = s

    if get_syms:
        return concat_ + data, num_filter_1x1, syms
    else:
        return concat_ + data, num_filter_1x1

def get_hjnet_preact(use_global_stats, fix_gamma=True):
    """ main shared conv layers """
    data = mx.sym.Variable(name='data')

    data_ = mx.sym.BatchNorm(data / 255.0, name='bn_data', fix_gamma=True, use_global_stats=use_global_stats)
    conv1_1 = mx.sym.Convolution(data_, name='conv1/1', num_filter=16,  
            kernel=(3,3), pad=(1,1), no_bias=True) # 32, 198
    conv1_2 = bn_crelu_conv(conv1_1, postfix_name='1/2', 
            num_filter=32, kernel=(3,3), pad=(1,1), 
            use_global_stats=use_global_stats, fix_gamma=fix_gamma) # 48, 196 
    conv1_3 = bn_crelu_conv_poolup2(conv1_2, postfix_name='1/3', 
            num_filter=64, kernel=(3,3), pad=(1,1), 
            use_global_stats=use_global_stats, fix_gamma=fix_gamma) # 48, 192 
    # crop1_2 = mx.sym.Crop(conv1_2, conv1_3, center_crop=True)
    concat1 = mx.sym.concat(conv1_2, conv1_3)

    nf_3x3 = [32, 40, 48, 32] # 24 48 96 192 384
    nf_1x1 = [32*3, 40*3, 48*3, 32*3]
    n_incep = [2, 2, 2, 2]

    group_i = pool(concat1, kernel=(2,2))
    groups = []
    n_curr_ch = 96
    for i in range(4):
        for j in range(n_incep[i]):
            group_i, n_curr_ch, syms = inception_group(group_i, 'g{}/u{}'.format(i+2, j+1), n_curr_ch, 
                    num_filter_3x3=nf_3x3[i], num_filter_1x1=nf_1x1[i],  
                    use_global_stats=use_global_stats, fix_gamma=fix_gamma, get_syms=True) 
        group_i = pool(group_i, kernel=(2,2), name='pool{}'.format(i+2)) # 96 48 24 12 6
        groups.append(group_i)

    # for context feature
    n_curr_ch = nf_1x1[-2]
    nf_3x3_ctx = 32
    nf_1x1_ctx = 32*3
    group_c = groups[-2]
    for i in range(2):
        group_c, n_curr_ch, s = inception_group(group_c, 'g_ctx/u{}'.format(i+1), n_curr_ch,
                num_filter_3x3=nf_3x3_ctx, num_filter_1x1=nf_1x1_ctx, 
                use_global_stats=use_global_stats, fix_gamma=fix_gamma, get_syms=True)
    group_c = pool(group_c, kernel=(2,2), name='pool_ctx')

    # upsample feature for small face (12px)
    conv0 = bn_relu_conv(groups[0], prefix_name='group0/', 
            num_filter=32, kernel=(1,1), 
            use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    bn0 = bn_relu(conv0, name='g0/bnu', use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    convu = mx.sym.Convolution(bn0, name='g0/convu', num_filter=128, kernel=(3,3), pad=(1,1), no_bias=True)
    convu = subpixel_upsample(convu, 32, 2, 2)

    groups = [convu] + groups
    return groups, group_c
