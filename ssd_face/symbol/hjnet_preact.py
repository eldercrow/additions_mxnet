# import find_mxnet
import mxnet as mx

def subpixel_upsample(data, ch, c, r):
    if r == 1 and c == 1:
        return data
    X = mx.sym.reshape(data=data, shape=(-3, 0, 0)) # (bsize*ch*r*c, a, b)
    X = mx.sym.reshape(data=X, shape=(-4, -1, r*c, 0, 0)) # (bsize*ch, r*c, a, b)
    X = mx.sym.transpose(data=X, axes=(0, 3, 2, 1)) # (bsize*ch, b, a, r*c)
    X = mx.sym.reshape(data=X, shape=(0, 0, -1, c)) # (bsize*ch, b, a*r, c)
    X = mx.sym.transpose(data=X, axes=(0, 2, 1, 3)) # (bsize*ch, a*r, b, c)
    X = mx.sym.reshape(data=X, shape=(-4, -1, ch, 0, -3)) # (bsize, ch, a*r, b*c)
    return X

def conv_subpixel(data, name, num_filter, 
        kernel=(3,3), dilate=(2,2), pad=(0,0), pool_type='max', no_bias=True):
    # no stride accepted, at least for now
    if dilate[0] == 1 and dilate[1] == 1: # ordinary conv
        return mx.sym.Convolution(data=data, name=name, num_filter=num_filter, 
                kernel=kernel, pad=pad, no_bias=no_bias)

    multiplier = dilate[0]*dilate[1]
    # pooling
    pool_ = pool(data, kernel=dilate, stride=dilate, pool_type=pool_type)
    # conv
    n_filter_pooled = num_filter * multiplier
    wd_mult = 1.0 / multiplier
    conv_ = mx.sym.Convolution(data=pool_, name=name, num_filter=n_filter_pooled, 
            attr={'__wd_mult__': str(wd_mult)}, 
            kernel=kernel, pad=(pad[0]/dilate[0], pad[1]/dilate[1]), no_bias=no_bias)
    # subpixel recover
    return subpixel_upsample(conv_, num_filter, dilate[0], dilate[1])

def bn_relu(data, name, use_global_stats=False, fix_gamma=True):
    bn_ = mx.sym.BatchNorm(data, use_global_stats=use_global_stats, fix_gamma=fix_gamma, name=name)
    relu_ = mx.sym.Activation(bn_, act_type='relu')
    return relu_

def downsample_conv(data, num_filter, name, pool_type='max', wd_mult=1.0, 
        kernel=(3,3), dilate=(1,1), stride=(1,1), pad=(0,0), no_bias=True):
    if dilate[0] == 1 and dilate[1] == 1: # ordinary conv
        conv_ = mx.sym.Convolution(data=data, name=name, num_filter=num_filter, 
                kernel=kernel, pad=pad, no_bias=no_bias)
    else:
        # pooling
        pool_ = pool(data, kernel=dilate, stride=dilate, pool_type=pool_type)
        # conv
        conv_ = mx.sym.Convolution(data=pool_, name=name, num_filter=num_filter, 
                attr={'__wd_mult__': str(wd_mult)}, 
                kernel=kernel, pad=(pad[0]/dilate[0], pad[1]/dilate[1]), no_bias=no_bias)
    return conv_

def bn_relu_conv(data, num_filter, prefix_name=None, postfix_name=None, 
        kernel=(3,3), dilate=(1,1), stride=(1,1), pad=(0,0), 
        use_global_stats=False, fix_gamma=True, no_bias=True):
    #
    if prefix_name is None and postfix_name is None:
        conv_name = None
        bn_name = None
    else:
        pr = prefix_name if prefix_name is not None else ''
        po = postfix_name if postfix_name is not None else ''
        conv_name = pr + 'conv' + po
        bn_name = pr + 'bn' + po

    bn_ = mx.sym.BatchNorm(data, use_global_stats=use_global_stats, fix_gamma=fix_gamma, name=bn_name)
    relu_ = mx.sym.Activation(bn_, act_type='relu')
    conv_ = conv_subpixel(relu_, name=conv_name, num_filter=num_filter, 
            kernel=kernel, dilate=dilate, pad=pad, pool_type='max', no_bias=no_bias)
    # conv_ = mx.sym.Convolution(data=data, num_filter=num_filter, 
    #         kernel=kernel, dilate=dilate, stride=stride, pad=pad, no_bias=True, name=conv_name)
    return conv_

def bn_crelu_conv(data, num_filter, prefix_name=None, postfix_name=None, 
        kernel=(3,3), dilate=(1,1), stride=(1,1), pad=(0,0), 
        use_global_stats=False, fix_gamma=True, no_bias=True): 
    #
    if prefix_name is None and postfix_name is None:
        conv_name = None
        bn_name = None
    else:
        pr = prefix_name if prefix_name is not None else ''
        po = postfix_name if postfix_name is not None else ''
        conv_name = pr + 'conv' + po
        bn_name = pr + 'bn' + po
        # sigma_name = pr + 'dn' + po + '_beta'

    bn_ = mx.sym.BatchNorm(data, use_global_stats=use_global_stats, fix_gamma=fix_gamma, name=bn_name)
    # bn_ = conv_norm(data, sigma_name, nch, kernel=kernel)
    concat_ = mx.sym.concat(bn_, -bn_)
    relu_ = mx.sym.Activation(concat_, act_type='relu')
    conv_ = conv_subpixel(relu_, name=conv_name, num_filter=num_filter, 
            kernel=kernel, dilate=dilate, pad=pad, pool_type='max', no_bias=no_bias)
    # conv_ = mx.sym.Convolution(data=data, num_filter=num_filter, 
    #         kernel=kernel, dilate=dilate, stride=stride, pad=pad, 
    #         no_bias=True, name=conv_name)
    return conv_

def pool(data, kernel=(2,2), stride=(2,2), pool_type='max'):
    pool_ = mx.sym.Pooling(data=data, kernel=kernel, stride=stride, pool_type=pool_type)
    return pool_

def inception_group(data, prefix_group_name, n_curr_ch, 
        num_filter_3x3, num_filter_1x1, dilates=((1,1),(2,2),(2,2)), 
        use_global_stats=False, fix_gamma=True):
    """ 
    inception unit, only full padding is supported
    """
    n_group = len(dilates)

    prefix_name = prefix_group_name + '/'
    incep_layers = []
    conv_ = data
    nf = num_filter_3x3
    df = [1, 1]
    for ii in range(n_group):
        nf *= dilates[ii][0] * dilates[ii][1]
        df[0] *= dilates[ii][0]
        df[1] *= dilates[ii][1]
        wd_mult = 1.0 / df[0] / df[1]
        postfix_name = '3x3/' + str(ii+1)
        # conv_ = bn_relu_conv(conv_, num_filter_3x3, prefix_name, postfix_name, 
        #         kernel=(3,3), dilate=dilates[ii], pad=dilates[ii])
        relu_ = bn_relu(conv_, name=prefix_name+'bn'+postfix_name, 
                use_global_stats=use_global_stats, fix_gamma=fix_gamma)
        conv_ = downsample_conv(relu_, nf, name=prefix_name+'conv'+postfix_name, wd_mult=wd_mult, 
                kernel=(3,3), dilate=dilates[ii], pad=dilates[ii])
        incep_layers.append(subpixel_upsample(conv_, num_filter_3x3, df[0], df[1]))

    concat_ = mx.sym.concat(*incep_layers)

    if (num_filter_3x3 * n_group) != num_filter_1x1:
        concat_ = bn_relu_conv(concat_, num_filter_1x1, prefix_name, '1x1', 
                kernel=(1,1), 
                use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    
    if n_curr_ch != num_filter_1x1:
        data = bn_relu_conv(data, num_filter_1x1, prefix_name+'proj/', 
                kernel=(1,1), 
                use_global_stats=use_global_stats, fix_gamma=fix_gamma)

    return concat_ + data, num_filter_1x1

def get_hjnet_preact(data, is_test, fix_gamma=True):
    """ main shared conv layers """
    data = mx.sym.Variable(name='data')

    data_ = mx.sym.BatchNorm(data / 255.0, name='bn_data', fix_gamma=True, use_global_stats=is_test)
    conv1_1 = mx.sym.Convolution(data_, name='conv1/1', num_filter=16,  
            kernel=(3,3), pad=(0,0), no_bias=True) # 32, 198
    conv1_2 = bn_crelu_conv(conv1_1, 32, postfix_name='1/2', 
            kernel=(3,3), pad=(0,0), use_global_stats=is_test, fix_gamma=fix_gamma) # 48, 196 
    conv1_3 = bn_crelu_conv(conv1_2, 64, postfix_name='1/3', 
            kernel=(3,3), dilate=(2,2), pad=(0,0), use_global_stats=is_test, fix_gamma=fix_gamma) # 48, 192 
    crop1_2 = mx.sym.Crop(conv1_2, conv1_3, center_crop=True)
    concat1 = mx.sym.concat(crop1_2, conv1_3)

    nf_3x3 = [40, 48, 56, 64] # 24 48 96 192 384
    nf_1x1 = [40*3, 48*3, 56*3, 64*3]
    dilates = [((1,1), (1,1), (2,2))] * 4
    n_incep = [2, 2, 2, 2]

    pool_i = pool(concat1, kernel=(2,2))
    groups = []
    n_curr_ch = 96
    for i in range(4):
        for j in range(n_incep[i]):
            group_i, n_curr_ch = inception_group(pool_i, 'group{}/unit{}'.format(i+2, j+1), n_curr_ch, 
                    num_filter_3x3=nf_3x3[i], num_filter_1x1=nf_1x1[i], dilates=dilates[i], 
                    use_global_stats=is_test, fix_gamma=fix_gamma) 
        pool_i = pool(group_i, kernel=(2,2), name='pool{}'.format(i+2)) # 96 48 24 12 6
        groups.append(pool_i)
    return groups
