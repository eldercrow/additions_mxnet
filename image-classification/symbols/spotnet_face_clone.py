import mxnet as mx
import numpy as np
from net_block_spotnet import *
from ast import literal_eval as make_tuple


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

    dilates = ((1, 1), (2, 2), (1, 1))

    bn_, s = relu_conv_bn(data, prefix_name=prefix_name + 'init/',
            num_filter=num_filter_3x3[0], kernel=(1, 1),
            use_global_stats=use_global_stats, get_syms=True)
    syms['init'] = s

    incep_layers = [bn_]
    for ii in range(3):
        bn_, s = relu_conv_bn(bn_, prefix_name=prefix_name + '3x3/{}/'.format(ii),
            num_filter=num_filter_3x3[ii], kernel=(3, 3), pad=dilates[ii], dilate=dilates[ii],
            use_global_stats=use_global_stats, get_syms=True)
        syms['unit{}'.format(ii)] = s
        incep_layers.append(bn_)
        bn_ = mx.sym.concat(*incep_layers)

    concat_, s = relu_conv_bn(bn_, prefix_name=prefix_name + 'concat/',
            num_filter=num_filter_1x1, kernel=(1, 1),
            use_global_stats=use_global_stats, get_syms=True)
    syms['concat'] = s

    if num_filter_1x1 != n_curr_ch:
        data, s = relu_conv_bn(data, prefix_name=prefix_name + 'proj/',
            num_filter=num_filter_1x1, kernel=(1, 1),
            use_global_stats=use_global_stats, get_syms=True)
        syms['proj_data'] = s

    res_ = concat_ + data

    if get_syms:
        return res_, num_filter_1x1, syms
    else:
        return res_, num_filter_1x1


def clone_inception_group(data, prefix_group_name, src_syms):
    """
    inception unit, only full padding is supported
    """
    prefix_name = prefix_group_name
    bn_ = clone_relu_conv_bn(data, prefix_name+'init/', src_syms['init'])

    incep_layers = [bn_]
    for ii in range(3):
        bn_ = clone_relu_conv_bn(bn_, prefix_name+'3x3/{}/'.format(ii), src_syms['unit{}'.format(ii)])
        incep_layers.append(bn_)
        bn_ = mx.sym.concat(*incep_layers)

    concat_ = clone_relu_conv_bn(bn_, prefix_name+'concat/', src_syms['concat'])

    if 'proj_data' in src_syms:
        data = clone_relu_conv_bn(data, prefix_name+'proj/', src_syms['proj_data'])
    return concat_ + data


def get_symbol(num_classes=1000, image_shape=(3, 224, 224), **kwargs):
    """ main shared conv layers """
    data = mx.sym.Variable(name='data')

    use_global_stats = False
    rf_ratio = 3
    if isinstance(image_shape, str):
        image_shape = make_tuple(image_shape)

    conv1 = mx.sym.Convolution(data / 128.0, name='1/conv',
            num_filter=16, kernel=(3, 3), pad=(1, 1), no_bias=True)  # 32, 198
    concat1 = mx.sym.concat(conv1, -conv1, name='concat1')
    bn1 = mx.sym.BatchNorm(concat1, name='1/bn', use_global_stats=use_global_stats, fix_gamma=False)
    pool1 = pool(bn1) # 112

    bn2 = relu_conv_bn(pool1, prefix_name='2/',
            num_filter=32, kernel=(3, 3), pad=(1, 1), use_crelu=True,
            use_global_stats=use_global_stats)

    curr_sz = 4 * rf_ratio

    n_curr_ch = 64
    nf_3x3 = [(16, 16, 16), (24, 24, 24), (32, 32, 32)] # 56 28 14
    nf_1x1 = [128, 192, 256]
    strides = [4, 8, 16]

    curr_sz *= 8
    while curr_sz <= max(image_shape) / 2:
        curr_sz *= 2
        strides.append(strides[-1] * 2)

    n_group = len(strides)
    nf_3x3_clone = (24, 24, 24) # (32, 16, 16)
    nf_1x1_clone = 128

    ''' basic groups
    '''
    group_i = bn2
    groups = []
    n_curr_ch = 64
    for i, (nf3, nf1) in enumerate(zip(nf_3x3, nf_1x1)):
        group_i = pool(group_i)
        group_i, n_curr_ch = inception_group(group_i, 'g{}/'.format(i), n_curr_ch,
                num_filter_3x3=nf3, num_filter_1x1=nf1,
                use_global_stats=use_global_stats, get_syms=False)
        groups.append(group_i)

    ''' buffer layer for constructing clone layers
    '''
    group_i = pool(groups[-1])
    clone_buffer = relu_conv_bn(group_i, prefix_name='clone_buffer/',
        num_filter=nf_1x1_clone, kernel=(1, 1), pad=(0, 0),
        use_global_stats=use_global_stats)

    # clone reference layer
    syms_clone = {}
    relu_clone = mx.sym.Activation(clone_buffer, act_type='relu')
    conv_clone = mx.sym.Convolution(relu_clone, name='g3/proj/clone/conv',
            num_filter=nf_1x1_clone, kernel=(1,1), pad=(0,0), no_bias=True)
    syms_clone['proj'] = conv_clone
    # group_i, sym_dn = data_norm(group_i, name='dn/3', nch=64, get_syms=True)
    norm_clone = mx.sym.InstanceNorm(conv_clone, name='g3/dn')
    syms_clone['norm'] = norm_clone

    n_curr_ch = nf_1x1_clone
    group_i, n_curr_ch, s = inception_group(norm_clone, 'g3/clone/', n_curr_ch,
            num_filter_3x3=nf_3x3_clone, num_filter_1x1=nf_1x1_clone,
            use_global_stats=use_global_stats, get_syms=True)
    syms_clone['incep'] = s
    groups.append(group_i)

    ''' cloned layers
    '''
    for i in range(4, n_group):
        group_cloned = pool(groups[-1])
        group_cloned = mx.sym.Activation(group_cloned, act_type='relu')
        group_cloned = clone_conv(group_cloned, 'g{}/proj/clone/conv'.format(i), syms_clone['proj'])
        group_cloned = clone_in(group_cloned, 'g{}/dn'.format(i), syms_clone['norm'])
        group_cloned = clone_inception_group(group_cloned, 'g{}/clone/'.format(i), syms_clone['incep'])
        groups.append(group_cloned)  # 192 384 768

    poolg = pool(groups[-1], name='poolg', kernel=(3, 3), stride=(2, 2), pool_type='avg')
    flatten = mx.sym.flatten(poolg)
    fc1 = mx.sym.FullyConnected(data=flatten, num_hidden=num_classes)
    softmax = mx.sym.SoftmaxOutput(data=fc1, name='softmax')
    return softmax
