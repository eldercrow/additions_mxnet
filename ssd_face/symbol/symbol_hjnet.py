# import find_mxnet
import mxnet as mx
import numpy as np
from common import multibox_layer_pvanet
from multibox_target import *

def conv_bn(data, num_filter, prefix_name=None, postfix_name=None, 
        kernel=(3,3), dilate=(1,1), stride=(1,1), pad=(0,0), 
        use_global_stats=False, fix_gamma=True):
    #
    if prefix_name is None and postfix_name is None:
        conv_name = None
        bn_name = None
    else:
        pr = prefix_name if prefix_name is not None else ''
        po = postfix_name if postfix_name is not None else ''
        conv_name = pr + 'conv' + po
        bn_name = pr + 'bn' + po

    conv_ = mx.symbol.Convolution(data=data, num_filter=num_filter, 
            kernel=kernel, dilate=dilate, stride=stride, pad=pad, no_bias=True, name=conv_name)
    bn_ = mx.symbol.BatchNorm(data=conv_, use_global_stats=use_global_stats, fix_gamma=fix_gamma, name=bn_name)
    return bn_ 

def conv_bn_relu(data, num_filter, prefix_name=None, postfix_name=None, 
        kernel=(3,3), dilate=(1,1), stride=(1,1), pad=(0,0), 
        use_global_stats=False, fix_gamma=True):
    #
    if prefix_name is None and postfix_name is None:
        conv_name = None
        bn_name = None
        relu_name = None
    else:
        pr = prefix_name if prefix_name is not None else ''
        po = postfix_name if postfix_name is not None else ''
        conv_name = pr + 'conv' + po
        bn_name = pr + 'bn' + po
        relu_name = pr + 'relu' + po

    conv_ = mx.symbol.Convolution(data=data, num_filter=num_filter, 
            kernel=kernel, dilate=dilate, stride=stride, pad=pad, no_bias=True, name=conv_name)
    bn_ = mx.symbol.BatchNorm(data=conv_, use_global_stats=use_global_stats, fix_gamma=fix_gamma, name=bn_name)
    relu_ = mx.symbol.Activation(data=bn_, act_type='relu', name=relu_name)
    return relu_

def cconv_bn_relu(data, num_filter, prefix_name=None, postfix_name=None, 
        kernel=(3,3), dilate=(1,1), stride=(1,1), pad=(0,0), 
        use_global_stats=False, fix_gamma=True): 
    #
    if prefix_name is None and postfix_name is None:
        conv_name = None
        cconv_name = None
        bn_name = None
        relu_name = None
    else:
        pr = prefix_name if prefix_name is not None else ''
        po = postfix_name if postfix_name is not None else ''
        conv_name = pr + 'conv' + po
        cconv_name = pr + 'cconv' + po
        bn_name = pr + 'bn' + po
        relu_name = pr + 'relu' + po

    conv_ = mx.symbol.Convolution(data=data, num_filter=num_filter, 
            kernel=kernel, dilate=dilate, stride=stride, pad=pad, 
            no_bias=True, name=conv_name)
    cconv_ = mx.symbol.Concat(conv_, -conv_, name=cconv_name)
    bn_ = mx.symbol.BatchNorm(data=cconv_, use_global_stats=use_global_stats, fix_gamma=fix_gamma, name=bn_name)
    relu_ = mx.symbol.Activation(data=bn_, act_type='relu', name=relu_name)
    return relu_

def fc_bn_relu(data, num_hidden, prefix_name=None, postfix_name=None, 
        use_global_stats=False, fix_gamma=True):
    #
    if prefix_name is None and postfix_name is None:
        fc_name = None
        bn_name = None
        relu_name = None
    else:
        pr = prefix_name if prefix_name is not None else ''
        po = postfix_name if postfix_name is not None else ''
        fc_name = pr + 'fc' + po
        bn_name = pr + 'bn' + po
        relu_name = pr + 'relu' + po

    fc_ = mx.symbol.FullyConnected(data=data, num_hidden=num_hidden, no_bias=True, name=fc_name)
    bn_ = mx.symbol.BatchNorm(data=fc_, use_global_stats=use_global_stats, fix_gamma=fix_gamma, name=bn_name)
    relu_ = mx.symbol.Activation(data=bn_, act_type='relu', name=relu_name)
    return relu_

def fc_relu(data, num_hidden, prefix_name=None, postfix_name=None):
    #
    if prefix_name is None and postfix_name is None:
        fc_name = None
    else:
        pr = prefix_name if prefix_name is not None else ''
        po = postfix_name if postfix_name is not None else ''
        fc_name = pr + 'fc' + po

    fc_ = mx.symbol.FullyConnected(data=data, num_hidden=num_hidden, name=fc_name)
    relu_ = mx.symbol.Activation(data=fc_, act_type='relu')
    return relu_

def pool(data, kernel=(2,2), stride=(2,2), pool_type='max'):
    pool_ = mx.symbol.Pooling(data=data, kernel=kernel, stride=stride, pool_type=pool_type)
    return pool_

def conv_dilate_group(data, prefix_group_name, n_curr_ch, 
        num_filter_3x3=16, num_filter_1x1=32, dilates=((1,1),(2,2),(2,2),(3,3)), pads=None, 
        pad_opt='all', use_global_stats=False):
    """ A group of conv kernels. """
    need_crop = True
    if pads is None or len(pads) == 0:
        if pad_opt=='none':
            pads = [(0,0)] * len(dilates)
            need_crop = True
        elif pad_opt=='all':
            pads = dilates
            need_crop = False
        else:
            assert False, 'Available pad_opt= {"none", "all"}'
    assert len(dilates) == len(pads)
    n_group = len(dilates)

    in_layer = data
    for ii in range(n_group):
        prefix_name = prefix_group_name + '/'
        postfix_name_3x3 = '%d/3x3' % (ii+1)
        postfix_name_1x1 = '%d/1x1' % (ii+1)
        # 3x3 conv
        conv3x3 = conv_bn_relu(in_layer, num_filter_3x3, prefix_name, postfix_name_3x3, 
                kernel=(3,3), dilate=dilates[ii], stride=(1,1), pad=pads[ii], 
                use_global_stats=use_global_stats)
        # 1x1 conv
        conv1x1 = conv_bn_relu(conv3x3, num_filter_1x1, prefix_name, postfix_name_1x1, 
                kernel=(1,1),
                use_global_stats=use_global_stats)

        need_res = (n_curr_ch == num_filter_1x1)
        need_crop = dilates[ii] != pads[ii]
        # residual
        if need_res:
            if need_crop:
                in_layer = conv1x1 + mx.symbol.Crop(in_layer, conv1x1, center_crop=True)
            else:
                in_layer = conv1x1 + in_layer
        else:
            in_layer = conv1x1
        n_curr_ch = num_filter_1x1

    return in_layer, n_curr_ch

def get_hjnet_conv(data, is_test):
    conv1_1 = cconv_bn_relu(data, 16, postfix_name='1/1', 
            kernel=(3,3), pad=(0,0), use_global_stats=is_test) # 32, 198
    conv1_2 = cconv_bn_relu(conv1_1, 24, postfix_name='1/2', 
            kernel=(3,3), pad=(0,0), use_global_stats=is_test) # 64, 196
    conv1_3 = cconv_bn_relu(conv1_2, 32, postfix_name='1/3', 
            kernel=(3,3), dilate=(2,2), pad=(0,0), use_global_stats=is_test) # 64, 192
    pooling = pool(conv1_3, kernel=(2,2)) # 96

    nf_3x3 = [16, 16, 32, 32, 64]
    nf_1x1 = [32, 48, 64, 96, 128]

    groups = [conv1_3]
    n_curr_ch = 64
    for i in range(5):
        pooling = pool(groups[i], kernel=(2,2))
        group_i, n_curr_ch = conv_dilate_group(pooling, 'group{}'.format(i+2), n_curr_ch, 
                num_filter_3x3=nf_3x3[i], num_filter_1x1=nf_1x1[i], use_global_stats=is_test) # [12, 24, 48, 96, 192]
        groups.append(group_i)

    return groups

def get_symbol_train(num_classes=1000, prefix_symbol='', **kwargs):
    """ PLEEEASEEEEEEEEEEE!!!! """
    is_test = False
    if 'is_test' in kwargs:
        is_test = kwargs['is_test']
    data = mx.sym.Variable('data') # 3, 200
    label = mx.sym.Variable('label')

    groups = get_hjnet_conv(data, is_test)
    from_layers = groups[2:]

    sizes = []
    sizes.append([1.0/8.0, np.sqrt(2.0)/16.0])
    sizes.append([1.0/4.0, np.sqrt(2.0)/8.0])
    sizes.append([1.0/2.0, np.sqrt(2.0)/4.0])
    sizes.append([1.0, np.sqrt(2.0)/2.0])

    # sizes = [[1.0 / 32.0, 1.0 / 16.0], [1.0 / 8.0], [1.0 / 4.0], [1.0 / 2.0], [1.0]]
    ratios = [1.0, 0.8, 1.25]
    normalization = -1
    clones = [] #[3, 4]
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
