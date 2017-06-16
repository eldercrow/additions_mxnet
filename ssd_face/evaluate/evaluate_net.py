import os
import sys
import importlib
# VOC
from dataset.pascal_voc import PascalVoc
from dataset.iterator import DetIter
from detect.detector import Detector
# Wider Face
from dataset.wider import Wider
from dataset.face_test_iter import FaceTestIter
from detect.face_detector import FaceDetector
from config.config import cfg
import logging


def evaluate_net(net,
                 dataset,
                 devkit_path,
                 mean_pixels,
                 data_shape,
                 model_prefix,
                 epoch,
                 ctx,
                 year=None,
                 sets='test',
                 batch_size=1,
                 th_pos=0.5,
                 nms_thresh=1.0/3.0,
                 force_nms=False):
    """
    Evaluate entire dataset, basically simple wrapper for detections

    Parameters:
    ---------
    dataset : str
        name of dataset to evaluate
    devkit_path : str
        root directory of dataset
    mean_pixels : tuple of float
        (R, G, B) mean pixel values
    data_shape : int
        resize input data shape, or maximum avaliable size for face detection
    model_prefix : str
        load model prefix
    epoch : int
        load model epoch
    ctx : mx.ctx
        running context, mx.cpu() or mx.gpu(0)...
    year : str or None
        evaluate on which year's data
    sets : str
        evaluation set
    batch_size : int
        using batch_size for evaluation
    nms_thresh : float
        non-maximum suppression threshold
    force_nms : bool
        force suppress different categories
    """
    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if dataset == "pascal":
        if not year:
            year = '2007'
        imdb = PascalVoc(
            sets, year, devkit_path, shuffle=False, is_train=False)
        data_iter = DetIter(
            imdb,
            batch_size,
            data_shape,
            mean_pixels,
            rand_samplers=[],
            rand_mirror=False,
            is_train=False,
            shuffle=False)
        sys.path.append(os.path.join(cfg.ROOT_DIR, 'symbol'))
        net = importlib.import_module("symbol_" + net) \
            .get_symbol(imdb.num_classes, nms_thresh, force_nms)
        model_prefix += "_" + str(data_shape)
        detector = Detector(net, model_prefix, epoch, data_shape, mean_pixels,
                            batch_size, ctx)
        logger.info("Start evaluation with {} images, be patient...".format(
            imdb.num_images))
        detections = detector.detect(data_iter)
        imdb.evaluate_detections(detections)
    elif dataset == 'wider':
        os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
        os.environ['MXNET_ENGINE_TYPE'] = 'NaiveEngine'
        max_data_shapes = (data_shape, data_shape)
        imdb = Wider(sets, devkit_path, shuffle=False, is_train=False)
        data_iter = FaceTestIter(imdb, max_data_shapes, mean_pixels)
        sys.path.append(os.path.join(cfg.ROOT_DIR, 'symbol'))
        net = importlib.import_module("symbol_" + net) \
            .get_symbol(imdb.num_classes, th_pos=th_pos, nms=nms_thresh)
        # model_prefix += "_" + str(data_shape)
        detector = FaceDetector(
            net, model_prefix, epoch, max_data_shapes, mean_pixels, ctx=ctx)
        logger.info("Start evaluation with {} images, be patient...".format(
            imdb.num_images))
        detections, im_paths = detector.detect(data_iter)
        # import ipdb
        # ipdb.set_trace()
        import cPickle as pickle
        with open('wider_eval_res_{}.pkl'.format(sets), 'wb') as fh:
            pickle.dump(detections, fh)
            pickle.dump(im_paths, fh)
    else:
        raise NotImplementedError("No support for dataset: " + dataset)
