import argparse
import mxnet as mx
import numpy as np
import json

from ast import literal_eval as make_tuple

import sys, os
sys.path.append(os.path.abspath('../layer'))
from multibox_prior_layer import *
from multibox_target_layer import *
from multibox_detection_layer import *
from smoothed_focal_loss_layer import *

def parse_args():
    #
    parser = argparse.ArgumentParser(description='MAC estimation tool for SSD')
    parser.add_argument('--prefix', dest='prefix', help='model prefix', type=str)
    parser.add_argument('--epoch', dest='epoch', help='model epoch', type=int)
    parser.add_argument('--data-shape', dest='data_shape', help='data shape', type=str)
    parser.add_argument('--label-shape', dest='label_shape', help='label shape', type=str)

    args = parser.parse_args()
    return args


def estimate_mac(prefix, epoch, data_shape, label_shape):
    '''
    '''
    net, _, _ = mx.model.load_checkpoint(prefix, epoch)
    layers = net.get_internals()
    arg_shapes, out_shapes, aux_shapes = layers.infer_shape_partial(data=data_shape, label=label_shape)

    json_net = json.loads(net.tojson())
    name_op_dict = {}
    for n in json_net['nodes']:
        nname = str(n['name'])
        nop = str(n['op'])
        name_op_dict[nname] = nop

    # map name to shape
    assert len(layers.list_outputs()) == len(out_shapes)
    dict_out_shapes = {}
    for n, s in zip(layers.list_outputs(), out_shapes):
        dict_out_shapes[n] = s
    dict_arg_shapes = {}
    for n, s in zip(layers.list_arguments(), arg_shapes):
        dict_arg_shapes[n] = s

    layer_info = {}
    for layer in layers:
        # process each layer
        # for now we estimate MACs from convolution, batchnorm and fc layers only.
        try:
            op_name = name_op_dict[layer.name].lower()
        except:
            continue
        else:
            if not op_name in ('convolution', 'batchnorm', 'fullyconnected'):
                continue
        if op_name == 'convolution':
            mac_num = _estimate_conv(layer, dict_out_shapes, dict_arg_shapes)
        if op_name == 'batchnorm':
            mac_num = _estimate_bn(layer, dict_out_shapes, dict_arg_shapes)
        if op_name == 'fullyconnected':
            mac_num = _estimate_fc(layer, dict_out_shapes, dict_arg_shapes)

        layer_info[layer.name] = (op_name, mac_num)

    total_mac = 0
    for _, v in layer_info.items():
        total_mac += v[1]

    import ipdb
    ipdb.set_trace()


def _estimate_conv(layer, out_shapes, arg_shapes):
    #
    out_name = layer.list_outputs()[0]
    oshape = out_shapes[out_name]
    wshape = (0, 0, 0, 0)
    bshape = (0,)

    for child in layer.get_children():
        if child.name.endswith('_weight'):
            wshape = arg_shapes[child.name]
        if child.name.endswith('_bias'):
            bshape = arg_shapes[child.name]

    oshape = np.array(oshape)
    oshape[1] = 1 # num channel is alread in wshape
    wshape = np.array(wshape)
    bshape = np.array(bshape)

    mac_num = np.prod(oshape) * (np.prod(wshape) + np.prod(bshape))
    return mac_num


def _estimate_bn(layer, out_shapes, arg_shapes):
    #
    out_name = layer.list_outputs()[0]
    oshape = out_shapes[out_name]
    gshape = (0,)
    bshape = (0,)

    for child in layer.get_children():
        if child.name.endswith('_gamma'):
            gshape = arg_shapes[child.name]
        if child.name.endswith('_beta'):
            bshape = arg_shapes[child.name]

    oshape = np.array(oshape)
    oshape[1] = 1 # num channel is alread in wshape
    gshape = np.array(gshape)
    bshape = np.array(bshape)

    mac_num = np.prod(oshape) * np.maximum(np.prod(gshape), np.prod(bshape))
    return mac_num


def _estimate_fc(layer, out_shapes, arg_shapes):
    #
    out_name = layer.list_outputs()[0]
    oshape = out_shapes[out_name]
    wshape = (0, 0)
    bshape = (0,)

    for child in layer.get_children():
        if child.name.endswith('_weight'):
            wshape = arg_shapes[child.name]
        if child.name.endswith('_bias'):
            bshape = arg_shapes[child.name]

    oshape = np.array(oshape)
    wshape = np.array(wshape)
    bshape = np.array(bshape)

    mac_num = oshape[0] * (np.prod(wshape) + np.prod(bshape))
    return mac_num


if __name__ == '__main__':
    #
    args = parse_args()
    if not args.prefix:
        args.prefix = '/home/hyunjoon/github/additions_mxnet/ssd/model/ssd_hypernetv6_384'
    if not args.epoch:
        args.epoch = 1000
    if not args.data_shape:
        args.data_shape = (1, 3, 384, 384)
    else:
        args.data_shape = make_tuple(args.data_shape)
    if not args.label_shape:
        args.label_shape = (1, 50, 6)
    else:
        args.label_shape = make_tuple(args.label_shape)

    estimate_mac(args.prefix, args.epoch, args.data_shape, args.label_shape)

