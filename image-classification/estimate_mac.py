import argparse
import mxnet as mx
import numpy as np
import json
import operator

from ast import literal_eval as make_tuple


def parse_args():
    #
    parser = argparse.ArgumentParser(description='MAC estimation tool for SSD')
    parser.add_argument('--prefix', dest='prefix', help='model prefix', type=str)
    parser.add_argument('--epoch', dest='epoch', help='model epoch', type=int)
    parser.add_argument('--data-shape', dest='data_shape', help='data shape', type=int)
    parser.add_argument('--label-shape', dest='label_shape', help='label shape', type=str)

    args = parser.parse_args()
    return args


def estimate_mac(net, data_shape, label_shape=None):
    '''
    '''
    # net, _, _ = mx.model.load_checkpoint(prefix, epoch)
    layers = net.get_internals()
    has_label = False
    for l in layers:
        if 'yolo_output_label' in l.name:
            has_label = True
            break

    data_shape = (1, 3, data_shape, data_shape)

    if has_label:
        arg_shapes, out_shapes, aux_shapes = \
                layers.infer_shape_partial(data=data_shape, yolo_output_label=label_shape)
    else:
        arg_shapes, out_shapes, aux_shapes = layers.infer_shape_partial(data=data_shape)

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

    layer_info = []
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

        flop_num = 0
        if op_name == 'convolution':
            flop_num = _estimate_conv(layer, dict_out_shapes, dict_arg_shapes)
        elif op_name == 'batchnorm':
            flop_num = _estimate_bn(layer, dict_out_shapes, dict_arg_shapes)
        # elif op_name == 'fullyconnected':
        #     flop_num = _estimate_fc(layer, dict_out_shapes, dict_arg_shapes)
        else:
            continue

        layer_info.append((layer.name, op_name, flop_num))

    total_mac = 0
    for v in layer_info:
        total_mac += v[2]
    total_gmac = np.round(total_mac / float(2**30), 3)

    for li in sorted(layer_info, key=operator.itemgetter(2)):
        lname = li[0]
        flop = li[2]
        gflop = np.round(flop / float(2**30), 3)
        print '{}: {:,} ({:,})'.format(lname, flop, gflop)
    print '-------------------------------------------------------'
    print 'Total {:,} ({:,}) MACs (GMACs) for conv, bn, fc layers.'.format(total_mac, total_gmac)
    print '-------------------------------------------------------'

    return total_mac, total_gmac


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

    flop_num = np.prod(oshape) * (np.prod(wshape) * 1 + np.prod(bshape))
    return flop_num


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

    flop_num = np.prod(oshape) * np.maximum(np.prod(gshape), np.prod(bshape))
    return flop_num


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

    flop_num = oshape[0] * (np.prod(wshape) * 1 + np.prod(bshape))
    return flop_num


if __name__ == '__main__':
    #
    import sys, os
    # from symbols.smoothed_focal_loss_layer import *
    # from symbols.smoothed_softmax_layer import *
    # from anchor_box_layer import *
    # from focal_loss_layer import *
    # from rpn_focal_loss_layer import *
    # from yolo_target_layer import *
    # from smoothed_focal_loss_layer import *
    # from roi_transform_layer import *
    # from iou_loss_layer import *

    args = parse_args()
    if not args.epoch:
        args.epoch = 1
    if not args.data_shape:
        args.data_shape = 416
    if not args.label_shape:
        args.label_shape = (1, 50, 6)
    else:
        args.label_shape = make_tuple(args.label_shape)

    try:
        net, _, _ = mx.model.load_checkpoint(args.prefix, args.epoch)
    except:
        net, _, _ = mx.model.load_checkpoint(args.prefix, 0)
    estimate_mac(net, args.data_shape, args.label_shape)
    # mx.viz.print_summary(net, {'data': (1, 3, 384, 384), 'label': (1, 50, 6)})

