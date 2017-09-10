from __future__ import print_function
import os
import sys
import importlib
import mxnet as mx
import numpy as np
from collections import OrderedDict
from dataset.face_test_iter import FaceTestIter
from config.config import cfg
from evaluate.eval_metric import MApMetric, VOC07MApMetric
import logging
from symbol.symbol_factory import get_symbol
from tools.do_nms import do_nms

def evaluate_net(net, imdb, mean_pixels, data_shape,
                 model_prefix, epoch, ctx=mx.cpu(), batch_size=1,
                 nms_thresh=0.45, force_nms=False,
                 ovp_thresh=0.5, use_difficult=False,
                 voc07_metric=False):
    """
    evalute network given validation record file

    Parameters:
    ----------
    net : str or None
        Network name or use None to load from json without modifying
    path_imgrec : str
        path to the record validation file
    path_imglist : str
        path to the list file to replace labels in record file, optional
    num_classes : int
        number of classes, not including background
    mean_pixels : tuple
        (mean_r, mean_g, mean_b)
    data_shape : tuple or int
        (3, height, width) or height/width
    model_prefix : str
        model prefix of saved checkpoint
    epoch : int
        load model epoch
    ctx : mx.ctx
        mx.gpu() or mx.cpu()
    batch_size : int
        validation batch size
    nms_thresh : float
        non-maximum suppression threshold
    force_nms : boolean
        whether suppress different class objects
    ovp_thresh : float
        AP overlap threshold for true/false postives
    use_difficult : boolean
        whether to use difficult objects in evaluation if applicable
    class_names : comma separated str
        class names in string, must correspond to num_classes if set
    voc07_metric : boolean
        whether to use 11-point evluation as in VOC07 competition
    """
    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    num_classes = imdb.num_classes
    class_names = imdb.classes

    # args
    if isinstance(data_shape, int):
        data_shape = (3, data_shape, data_shape)
    assert len(data_shape) == 3 and data_shape[0] == 3
    model_prefix += '_' + str(data_shape[1])

    # iterator
    eval_iter = FaceTestIter(imdb, mean_pixels, img_stride=128, fix_hw=True)
    # model params
    load_net, args, auxs = mx.model.load_checkpoint(model_prefix, epoch)
    # network
    if net is None:
        net = load_net
    else:
        net = get_symbol(net, data_shape[1], num_classes=num_classes,
            nms_thresh=nms_thresh, force_suppress=force_nms)
    if not 'label' in net.list_arguments():
        label = mx.sym.Variable(name='label')
        net = mx.sym.Group([net, label])

    # init module
    mod = mx.mod.Module(net, label_names=('label',), logger=logger, context=ctx,
        fixed_param_names=net.list_arguments())
    mod.bind(data_shapes=eval_iter.provide_data, label_shapes=eval_iter.provide_label)
    mod.set_params(args, auxs, allow_missing=False, force_init=True)

    # run evaluation
    if voc07_metric:
        metric = VOC07MApMetric(ovp_thresh, use_difficult, class_names)
    else:
        metric = MApMetric(ovp_thresh, use_difficult, class_names)

    results = []
    for i, (datum, im_info) in enumerate(eval_iter):
        mod.reshape(data_shapes=datum.provide_data, label_shapes=datum.provide_label)
        mod.forward(datum)

        preds = mod.get_outputs()

        det0 = preds[0][0].asnumpy() # (n_anchor, 6)
        det0 = do_nms(det0, 1, nms_thresh)
        preds[0][0] = mx.nd.array(det0, ctx=preds[0].context)

        sy, sx, _ = im_info['im_shape']
        scaler = mx.nd.array((1.0, sx, sy, sx, sy, 1.0))
        scaler = mx.nd.reshape(scaler, (1, 1, -1))

        datum.label[0] *= scaler
        metric.update(datum.label, preds)

        if i % 10 == 0:
            print('processed {} images.'.format(i))
        # if i == 10:
        #     break

    results = metric.get_name_value()
    for k, v in results:
        print("{}: {}".format(k, v))


def _do_nms(dets, th_nms):
    #
    areas = (dets[:, 4] - dets[:, 2]) * (dets[:, 5] - dets[:, 3])
    vmask = np.ones((dets.shape[0],), dtype=int)
    vidx = []
    for i, d in enumerate(dets):
        if vmask[i] == 0:
            continue
        iw = np.minimum(d[4], dets[i:, 4]) - np.maximum(d[2], dets[i:, 2])
        ih = np.minimum(d[5], dets[i:, 5]) - np.maximum(d[3], dets[i:, 3])
        I = np.maximum(iw, 0) * np.maximum(ih, 0)
        iou = I / np.maximum(areas[i:] + areas[i] - I, 1e-08)
        nidx = np.where(iou > th_nms)[0] + i
        vmask[nidx] = 0
        vidx.append(i)
    return vidx
