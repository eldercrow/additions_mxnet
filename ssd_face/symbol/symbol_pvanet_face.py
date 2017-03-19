import mxnet as mx
from net_block import *
from common import multibox_layer_pvanet
import numpy as np
from multibox_target import *

def mCReLU(data, group_name, filters, strides, use_global, n_curr_ch):
    """ 
    """
    kernels = ((1,1), (3,3), (1,1))
    pads = ((0,0), (1,1), (0,0))

    conv1, _ = bn_relu_conv(data=data, group_name=group_name+'_1', 
            num_filter=filters[0], pad=pads[0], kernel=kernels[0], stride=strides[0], use_global=use_global)
    conv2, _ = bn_relu_conv(data=conv1, group_name=group_name+'_2', 
            num_filter=filters[1], pad=pads[1], kernel=kernels[1], stride=strides[1], use_global=use_global)
    conv3 = bn_crelu_conv(data=conv2, group_name=group_name+'_3', 
            num_filter=filters[2], pad=pads[2], kernel=kernels[2], stride=strides[2], use_global=use_global)

    ss = 1
    for s in strides:
        ss *= s[0]
    need_proj = (n_curr_ch != filters[2]) or (ss != 1)
    if need_proj:
        proj = mx.symbol.Convolution(name=group_name+'_proj', data=data, 
                num_filter=filters[2], pad=(0,0), kernel=(1,1), stride=(ss,ss))
        res = conv3 + proj
    else:
        res = conv3 + data

    return res, filters[2]

# final_bn is to handle stupid redundancy in the original model
def inception(data, group_name, 
        filter_1, filters_3, filters_5, filter_p, filter_out, stride, use_global, n_curr_ch, final_bn=False):
    """
    """
    group_name = group_name + '_incep'

    group_name_1 = group_name + '_0'
    group_name_3 = group_name + '_1'
    group_name_5 = group_name + '_2'

    layer_syms = {}
    incep_bn = mx.symbol.BatchNorm(name=group_name+'_bn', data=data, 
        use_global_stats=use_global, fix_gamma=False)
    layer_syms[incep_bn.name] = incep_bn
    incep_relu = mx.symbol.Activation(name=group_name+'_relu', data=incep_bn, act_type='relu')

    incep_0, layers = conv_bn_relu(data=incep_relu, group_name=group_name_1, 
            num_filter=filter_1, kernel=(1,1), pad=(0,0), stride=stride, use_global=use_global)
    layer_syms.update(layers)

    incep_1_reduce, layers = conv_bn_relu(data=incep_relu, group_name=group_name_3+'_reduce', 
            num_filter=filters_3[0], kernel=(1,1), pad=(0,0), stride=stride, use_global=use_global)
    layer_syms.update(layers)

    incep_1_0, layers = conv_bn_relu(data=incep_1_reduce, group_name=group_name_3+'_0', 
            num_filter=filters_3[1], kernel=(3,3), pad=(1,1), stride=(1,1), use_global=use_global)
    layer_syms.update(layers)

    incep_2_reduce, layers = conv_bn_relu(data=incep_relu, group_name=group_name_5+'_reduce', 
            num_filter=filters_5[0], kernel=(1,1), pad=(0,0), stride=stride, use_global=use_global)
    layer_syms.update(layers)

    incep_2_0, layers = conv_bn_relu(data=incep_2_reduce, group_name=group_name_5+'_0', 
            num_filter=filters_5[1], kernel=(3,3), pad=(1,1), stride=(1,1), use_global=use_global)
    layer_syms.update(layers)

    incep_2_1, layers = conv_bn_relu(data=incep_2_0, group_name=group_name_5+'_1', 
            num_filter=filters_5[2], kernel=(3,3), pad=(1,1), stride=(1,1), use_global=use_global)
    layer_syms.update(layers)

    incep_layers = [incep_0, incep_1_0, incep_2_1]
    # incep_layers = [incep_0, incep_2_1]

    if filter_p is not None:
        incep_p_pool = mx.symbol.Pooling(name=group_name+'_pool', data=incep_relu, pooling_convention='full', 
                pad=(0,0), kernel=(3,3), stride=(2,2), pool_type='max')
        incep_p_proj, layers = conv_bn_relu(data=incep_p_pool, group_name=group_name+'_poolproj', 
                num_filter=filter_p, kernel=(1,1), pad=(0,0), stride=(1,1), use_global=use_global)
        layer_syms.update(layers)
        incep_layers.append(incep_p_proj)

    incep = mx.symbol.Concat(name=group_name, *incep_layers)
    out_conv = mx.symbol.Convolution(name=group_name.replace('_incep', '_out_conv'), data=incep, 
            num_filter=filter_out, kernel=(1,1), stride=(1,1), pad=(0,0))
    layer_syms[out_conv.name] = out_conv

    # final_bn is to handle stupid redundancy in the original model
    if final_bn:
        out_conv = mx.symbol.BatchNorm(name=group_name.replace('_incep', '_out_bn'), data=out_conv, 
                use_global_stats=use_global, fix_gamma=False)
    
    if n_curr_ch != filter_out or stride[0] > 1:
        out_proj = mx.symbol.Convolution(name=group_name.replace('_incep', '_proj'), data=data, 
                num_filter=filter_out, kernel=(1,1), stride=stride, pad=(0,0))
        layer_syms[out_proj.name] = out_proj
        res = out_conv + out_proj
    else:
        res = out_conv + data

    return res, filter_out, layer_syms

def clone_inception(data, group_name, src_syms, src_name):
    """
    """
    group_name = group_name + '_incep'

    group_name_1 = group_name + '_0'
    group_name_3 = group_name + '_1'
    group_name_5 = group_name + '_2'

    # name of layers to be cloned
    src_name = src_name + '_incep'
    src_names = {}
    src_names['incep_bn'] = src_name + '_bn'
    src_names['incep_0'] = src_name + '_0'
    src_names['incep_1_reduce'] = src_name + '_1_reduce'
    src_names['incep_1_0'] = src_name + '_1_0'
    src_names['incep_2_reduce'] = src_name + '_2_reduce'
    src_names['incep_2_0'] = src_name + '_2_0'
    src_names['incep_2_1'] = src_name + '_2_1'
    src_names['incep_p_proj'] = src_name + '_poolproj'
    src_names['out_conv'] = src_name.replace('_incep', '_out_conv')
    src_names['out_proj'] = src_name.replace('_incep', '_proj')

    layer_syms = {}
    # clone each conv_bn_relu layer
    incep_bn = clone_bn(data, group_name+'_bn', src_syms[src_names['incep_bn']])
    layer_syms[incep_bn.name] = incep_bn
    incep_relu = mx.symbol.Activation(name=group_name+'_relu', data=incep_bn, act_type='relu')

    incep_0, layers = clone_conv_bn_relu(incep_relu, group_name_1, src_syms, src_names['incep_0'])
    layer_syms.update(layers)

    incep_1_reduce, layers = clone_conv_bn_relu(
            incep_relu, group_name_3+'_reduce', src_syms, src_names['incep_1_reduce'])
    layer_syms.update(layers)
    incep_1_0, layers = clone_conv_bn_relu(
            incep_1_reduce, group_name_3+'_0', src_syms, src_names['incep_1_0'])
    layer_syms.update(layers)

    incep_2_reduce, layers = clone_conv_bn_relu(
            incep_relu, group_name_5+'_reduce', src_syms, src_names['incep_2_reduce'])
    layer_syms.update(layers)
    incep_2_0, layers = clone_conv_bn_relu(
            incep_2_reduce, group_name_5+'_0', src_syms, src_names['incep_2_0'])
    layer_syms.update(layers)
    incep_2_1, layers = clone_conv_bn_relu(
            incep_2_0, group_name_5+'_1', src_syms, src_names['incep_2_1'])
    layer_syms.update(layers)

    incep_layers = [incep_0, incep_1_0, incep_2_1]

    if src_names['incep_p_proj']+'_conv' in src_syms:
        incep_p_pool = mx.symbol.Pooling(name=group_name+'_pool', data=incep_relu, pooling_convention='full', 
                pad=(0,0), kernel=(3,3), stride=(2,2), pool_type='max')
        incep_p_proj, layers = clone_conv_bn_relu(
                incep_p_pool, group_name+'_poolproj', src_syms, src_names['incep_p_proj'])
        layer_syms.update(layers)
        incep_layers.append(incep_p_proj)

    incep = mx.symbol.Concat(name=group_name, *incep_layers)
    out_conv = clone_conv(incep, group_name.replace('_incep', '_out_conv'), src_syms[src_names['out_conv']])
    layer_syms[out_conv.name] = out_conv

    if src_names['out_proj'] in src_syms:
        out_proj = clone_conv(data, group_name.replace('_incep', '_proj'), src_syms[src_names['out_proj']])
        layer_syms[out_proj.name] = out_proj
        res = out_conv + out_proj
    else:
        res = out_conv + data

    return res, layer_syms

def pvanet_preact(is_test=False):
    """ PVANet 9.0 """
    out_layers = {}

    data = mx.symbol.Variable(name='data')
    conv1_1_conv = mx.symbol.Convolution(name='conv1_1_conv', data=data, 
            num_filter=16, pad=(3,3), kernel=(7,7), stride=(2,2), no_bias=True)
    conv1_1_concat = mx.symbol.Concat(name='conv1_1_concat', *[conv1_1_conv, -conv1_1_conv])
    conv1_1_bn = mx.symbol.BatchNorm(name='conv1_1_bn', data=conv1_1_concat, 
            use_global_stats=True, fix_gamma=False)
    conv1_1_relu = mx.symbol.Activation(name='conv1_1_relu', data=conv1_1_bn, act_type='relu')
    pool1 = mx.symbol.Pooling(name='pool1', data=conv1_1_relu, 
            pooling_convention='full', pad=(0,0), kernel=(3,3), stride=(2,2), pool_type='max')
    
    # no pre bn-scale-relu for 2_1_1
    conv2_1_1_conv = mx.symbol.Convolution(name='conv2_1_1_conv', data=pool1, 
            num_filter=24, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
    conv2_1_2_conv, _ = bn_relu_conv(data=conv2_1_1_conv, group_name='conv2_1_2', 
            num_filter=24, kernel=(3,3), pad=(1,1), stride=(1,1), use_global=True)
    conv2_1_3_conv = bn_crelu_conv(data=conv2_1_2_conv, group_name='conv2_1_3', 
            num_filter=64, kernel=(1,1), pad=(0,0), stride=(1,1), use_global=True)
    conv2_1_proj = mx.symbol.Convolution(name='conv2_1_proj', data=pool1, 
            num_filter=64, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
    conv2_1 = conv2_1_3_conv + conv2_1_proj

    # stack up mCReLU layers
    n_curr_ch = 64
    conv2_2, n_curr_ch = mCReLU(data=conv2_1, group_name='conv2_2', 
            filters=(24, 24, 64), strides=((1,1),(1,1),(1,1)), use_global=True, n_curr_ch=n_curr_ch)
    # rf 24
    conv2_3, n_curr_ch = mCReLU(data=conv2_2, group_name='conv2_3', 
            filters=(24, 24, 64), strides=((1,1),(1,1),(1,1)), use_global=True, n_curr_ch=n_curr_ch)
    out_layers['rf24'] = conv2_3

    conv3_1, n_curr_ch = mCReLU(data=conv2_3, group_name='conv3_1', 
            filters=(48, 48, 128), strides=((2,2),(1,1),(1,1)), use_global=True, n_curr_ch=n_curr_ch)
    conv3_2, n_curr_ch = mCReLU(data=conv3_1, group_name='conv3_2', 
            filters=(48, 48, 128), strides=((1,1),(1,1),(1,1)), use_global=True, n_curr_ch=n_curr_ch)
    conv3_3, n_curr_ch = mCReLU(data=conv3_2, group_name='conv3_3', 
            filters=(48, 48, 128), strides=((1,1),(1,1),(1,1)), use_global=True, n_curr_ch=n_curr_ch)
    # rf 48
    conv3_4, n_curr_ch = mCReLU(data=conv3_3, group_name='conv3_4', 
            filters=(48, 48, 128), strides=((1,1),(1,1),(1,1)), use_global=True, n_curr_ch=n_curr_ch) 
    out_layers['rf48'] = conv3_4

    # stack up inception layers
    conv4_1, n_curr_ch, names4_1 = inception(data=conv3_4, group_name='conv4_1', 
            filter_1=64, filters_3=(48,128), filters_5=(24,48,48), filter_p=128, filter_out=256, 
            stride=(2,2), use_global=True, n_curr_ch=n_curr_ch)
    conv4_2, n_curr_ch, names4_2 = inception(data=conv4_1, group_name='conv4_2', 
            filter_1=64, filters_3=(64,128), filters_5=(24,48,48), filter_p=None, filter_out=256, 
            stride=(1,1), use_global=True, n_curr_ch=n_curr_ch)
    conv4_3, n_curr_ch, names4_3 = inception(data=conv4_2, group_name='conv4_3', 
            filter_1=64, filters_3=(64,128), filters_5=(24,48,48), filter_p=None, filter_out=256, 
            stride=(1,1), use_global=True, n_curr_ch=n_curr_ch)
    conv4_4, n_curr_ch, names4_4 = inception(data=conv4_3, group_name='conv4_4', 
            filter_1=64, filters_3=(64,128), filters_5=(24,48,48), filter_p=None, filter_out=256, 
            stride=(1,1), use_global=True, n_curr_ch=n_curr_ch) # rf 96
    out_layers['rf96'] = conv4_4

    conv5_1, n_curr_ch, names5_1 = inception(data=conv4_4, group_name='conv5_1', 
            filter_1=64, filters_3=(96,192), filters_5=(32,64,64), filter_p=128, filter_out=384, 
            stride=(2,2), use_global=True, n_curr_ch=n_curr_ch)
    conv5_2, n_curr_ch, names5_2 = inception(data=conv5_1, group_name='conv5_2', 
            filter_1=64, filters_3=(96,192), filters_5=(32,64,64), filter_p=None, filter_out=384, 
            stride=(1,1), use_global=True, n_curr_ch=n_curr_ch)
    conv5_3, n_curr_ch, names5_3 = inception(data=conv5_2, group_name='conv5_3', 
            filter_1=64, filters_3=(96,192), filters_5=(32,64,64), filter_p=None, filter_out=384, 
            stride=(1,1), use_global=True, n_curr_ch=n_curr_ch)
    conv5_4, n_curr_ch, names5_4 = inception(data=conv5_3, group_name='conv5_4', 
            filter_1=64, filters_3=(96,192), filters_5=(32,64,64), filter_p=None, filter_out=384, 
            stride=(1,1), use_global=True, n_curr_ch=n_curr_ch) # rf 192
    out_layers['rf192'] = conv5_4

    # recursive inception layers - repeat incep5.
    conv6_0, names6_0 = relu_conv(data=conv5_4, group_name='conv6_0', num_filter=256, 
            kernel=(1,1), pad=(0,0), stride=(1,1), no_bias=True)

    conv6_1, names6_1 = clone_inception(conv6_0, 'conv6_1', names5_1, 'conv5_1')
    conv6_2, names6_2 = clone_inception(conv6_1, 'conv6_2', names5_2, 'conv5_2')
    conv6_3, names6_3 = clone_inception(conv6_2, 'conv6_3', names5_3, 'conv5_3')
    conv6_4, names6_4 = clone_inception(conv6_3, 'conv6_4', names5_4, 'conv5_4') # rf 384
    # conv6_1, n_curr_ch, names6_1 = inception(data=conv6_0, group_name='conv6_1', 
    #         filter_1=64, filters_3=(96,192), filters_5=(32,64,64), filter_p=128, filter_out=384, 
    #         stride=(2,2), use_global=True, n_curr_ch=n_curr_ch)
    # conv6_2, n_curr_ch, names6_2 = inception(data=conv6_1, group_name='conv6_2', 
    #         filter_1=64, filters_3=(96,192), filters_5=(32,64,64), filter_p=None, filter_out=384, 
    #         stride=(1,1), use_global=True, n_curr_ch=n_curr_ch)
    # conv6_3, n_curr_ch, names6_3 = inception(data=conv6_2, group_name='conv6_3', 
    #         filter_1=64, filters_3=(96,192), filters_5=(32,64,64), filter_p=None, filter_out=384, 
    #         stride=(1,1), use_global=True, n_curr_ch=n_curr_ch)
    # conv6_4, n_curr_ch, names6_4 = inception(data=conv6_3, group_name='conv6_4', 
    #         filter_1=64, filters_3=(96,192), filters_5=(32,64,64), filter_p=None, filter_out=384, 
    #         stride=(1,1), use_global=True, n_curr_ch=n_curr_ch) # rf 192
    out_layers['rf384'] = conv6_4

    # conv7_0, names7_0 = clone_relu_conv(conv6_4, 'conv7_0', names6_0, 'conv6_0')
    # conv7_1, names7_1 = clone_inception(conv7_0, 'conv7_1', names6_1, 'conv6_1')
    # conv7_2, names7_2 = clone_inception(conv7_1, 'conv7_2', names6_2, 'conv6_2')
    # conv7_3, names7_3 = clone_inception(conv7_2, 'conv7_3', names6_3, 'conv6_3')
    # conv7_4, names7_4 = clone_inception(conv7_3, 'conv7_4', names6_4, 'conv6_4') # rf 768
    # out_layers['rf768'] = conv7_4

    return out_layers

def build_hyperfeature(data, name, ctx_data, num_filter_proj, scale, num_filter_hyper):
    """
    """
    ctx_proj, _ = relu_conv(data=ctx_data, group_name='proj'+name, num_filter=num_filter_proj,
            kernel=(3,3), pad=(1,1), stride=(1,1), no_bias=True)
    ctx_up = mx.symbol.UpSampling(ctx_proj, num_args=1, name='up'+name, scale=scale, sample_type='nearest')
    concat = mx.symbol.Concat(data, ctx_up)
    hyper, _ = relu_conv(data=concat, group_name='hyper'+name, num_filter=num_filter_hyper,
            kernel=(1,1), pad=(0,0), stride=(1,1), no_bias=True)

    return hyper

def get_symbol_train(num_classes, **kwargs):
    """
    """
    out_layers = pvanet_preact(is_test=False)
    label = mx.symbol.Variable(name="label")

    # build hyperfeatures
    # 24 + 192
    hyper24 = build_hyperfeature(out_layers['rf24'], '24', out_layers['rf192'], 
            num_filter_proj=64, scale=8, num_filter_hyper=128)
    # 48 + 192
    hyper48 = build_hyperfeature(out_layers['rf48'], '48', out_layers['rf192'], 
            num_filter_proj=64, scale=4, num_filter_hyper=128)
    # 96 + 192
    hyper96 = build_hyperfeature(out_layers['rf96'], '96', out_layers['rf192'], 
            num_filter_proj=64, scale=2, num_filter_hyper=128)

    # project 192+ layers
    proj192, sym_proj192 = relu_conv(data=out_layers['rf192'], group_name='proj192', num_filter=64, 
            kernel=(3,3), pad=(1,1), stride=(1,1), no_bias=True)
    conv192, sym_conv192 = relu_conv(data=proj192, group_name='conv192', num_filter=128, 
            kernel=(1,1), pad=(1,1), stride=(1,1), no_bias=True)

    # proj384, sym_proj384 = relu_conv(data=out_layers['rf384'], group_name='proj384', num_filter=64, 
    #         kernel=(3,3), pad=(1,1), stride=(1,1), no_bias=True)
    # conv384, sym_conv384 = relu_conv(data=proj384, group_name='conv384', num_filter=128, 
    #         kernel=(1,1), pad=(1,1), stride=(1,1), no_bias=True)
    proj384, sym_proj384 = clone_relu_conv(data=out_layers['rf384'], group_name='proj384', 
            src_syms=sym_proj192, src_name='proj192')
    conv384, sym_conv384 = clone_relu_conv(data=proj384, group_name='conv384', 
            src_syms=sym_conv192, src_name='conv192')

    # proj768, sym_proj768 = clone_bn_relu_conv(data=out_layers['rf768'], group_name='proj768', 
    #         src_syms=sym_proj192, src_name='proj192')
    # conv768, sym_conv768 = clone_bn_relu_conv(data=proj768, group_name='conv768', 
    #         src_syms=sym_conv192, src_name='conv192')

    from_layers = [hyper24, hyper48, hyper96, conv192, conv384]
    sizes = [[1.0/16.0, 1.0/32.0, np.sqrt(2.0)/32.0]]
    sizes.append([1.0/8.0, np.sqrt(2.0)/16.0])
    sizes.append([1.0/4.0, np.sqrt(2.0)/8.0])
    sizes.append([1.0/2.0, np.sqrt(2.0)/4.0])
    sizes.append([1.0, np.sqrt(2.0)/2.0])

    # sizes = [[1.0 / 32.0, 1.0 / 16.0], [1.0 / 8.0], [1.0 / 4.0], [1.0 / 2.0], [1.0]]
    ratios = [1.0, 0.8, 1.25]
    normalization = -1
    clones = [3, 4]
    clip=True

    loc_preds, cls_preds, anchor_boxes = multibox_layer_pvanet(from_layers, \
            num_classes, is_test=False, sizes=sizes, ratios=ratios, clones=clones, clip=clip)
    tmp = mx.symbol.Custom(
            *[anchor_boxes, label, cls_preds], n_class=1, name='multibox_target', op_type='multibox_target')

    # tmp = mx.symbol.MultiBoxTarget(
    #     *[anchor_boxes, label, cls_preds], overlap_threshold=.5, \
    #     ignore_label=-1, negative_mining_ratio=3, minimum_negative_samples=0, \
    #     negative_mining_thresh=.5, variances=(0.1, 0.1, 0.2, 0.2),
    #     name="multibox_target")
    loc_target = tmp[0]
    loc_target_mask = tmp[1]
    cls_target = tmp[2]
    cls_preds = mx.symbol.transpose(cls_preds, axes=(0, 2, 1))

    cls_prob = mx.symbol.SoftmaxOutput(data=cls_preds, label=cls_target, \
        ignore_label=-1, use_ignore=True, grad_scale=3., multi_output=True, \
        normalization='batch', name="cls_prob")
    # cls_prob = mx.symbol.SoftmaxOutput(data=cls_preds, label=cls_target, \
    #     ignore_label=-1, use_ignore=True, grad_scale=3., multi_output=True, \
    #     normalization='valid', name="cls_prob")
    loc_diff = loc_preds - loc_target
    masked_loc_diff = loc_target_mask * loc_diff
    loc_loss_ = mx.symbol.smooth_l1(name="loc_loss_", \
        data=masked_loc_diff, scalar=1.0)
    loc_loss = mx.symbol.MakeLoss(loc_loss_, grad_scale=0.1, \
        normalization='batch', name="loc_loss")

    # monitoring training status
    cls_label = mx.symbol.MakeLoss(data=cls_target, grad_scale=0, name="cls_label")

    # group output
    out = mx.symbol.Group([cls_prob, loc_loss, cls_label])
    return out

# def get_symbol(num_classes, **kwargs):
#     """
#     """
#     out_layers = pvanet_preact(is_test=True)
#     return out_layers['rf768']
#
#
# if __name__ == '__main__':
#     net = get_symbol_train(1)
#
#     mod = mx.mod.Module(net)
#     mod.bind(data_shapes=[('data', (1, 3, 768, 768))])
#
#     mod.init_params()
#     arg_params, aux_params = mod.get_params()
#
#     for k in sorted(arg_params):
#         print k + ': ' + str(arg_params[k].shape)
#
#     import ipdb
#     ipdb.set_trace()
#
#     print 'done'
#
