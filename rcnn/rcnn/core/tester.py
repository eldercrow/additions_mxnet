# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import cPickle
import os
import time
import mxnet as mx
import numpy as np

from module import MutableModule
from rcnn.logger import logger
from rcnn.config import config
from rcnn.io import image
from rcnn.processing.bbox_transform import bbox_pred, clip_boxes
from rcnn.processing.part_transform import pred_head, pred_joint
from rcnn.processing.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper


class Predictor(object):
    def __init__(self, symbol, data_names, label_names,
                 context=mx.cpu(), max_data_shapes=None,
                 provide_data=None, provide_label=None,
                 arg_params=None, aux_params=None):
        self._mod = MutableModule(symbol, data_names, label_names,
                                  context=context, max_data_shapes=max_data_shapes)
        self._mod.bind(provide_data, provide_label, for_training=False)
        self._mod.init_params(arg_params=arg_params, aux_params=aux_params)

    def predict(self, data_batch):
        self._mod.forward(data_batch)
        return dict(zip(self._mod.output_names, self._mod.get_outputs()))


def im_proposal(predictor, data_batch, data_names, scale):
    data_dict = dict(zip(data_names, data_batch.data))
    output = predictor.predict(data_batch)

    # drop the batch index
    boxes = output['rois_output'].asnumpy()[:, 1:]
    scores = output['rois_score'].asnumpy()

    # transform to original scale
    boxes = boxes / scale

    return scores, boxes, data_dict


def generate_proposals(predictor, test_data, imdb, vis=False, thresh=0.):
    """
    Generate detections results using RPN.
    :param predictor: Predictor
    :param test_data: data iterator, must be non-shuffled
    :param imdb: image database
    :param vis: controls visualization
    :param thresh: thresh for valid detections
    :return: list of detected boxes
    """
    assert vis or not test_data.shuffle
    data_names = [k[0] for k in test_data.provide_data]

    i = 0
    t = time.time()
    imdb_boxes = list()
    original_boxes = list()
    for im_info, data_batch in test_data:
        t1 = time.time() - t
        t = time.time()

        scale = im_info[0, 2]
        scores, boxes, data_dict = im_proposal(predictor, data_batch, data_names, scale)
        t2 = time.time() - t
        t = time.time()

        # assemble proposals
        dets = np.hstack((boxes, scores))
        original_boxes.append(dets)

        # filter proposals
        keep = np.where(dets[:, 4:] > thresh)[0]
        dets = dets[keep, :]
        imdb_boxes.append(dets)

        if vis:
            vis_all_detection(data_dict['data'].asnumpy(), [dets], ['obj'], scale)

        logger.info('generating %d/%d ' % (i + 1, imdb.num_images) +
                    'proposal %d ' % (dets.shape[0]) +
                    'data %.4fs net %.4fs' % (t1, t2))
        i += 1

    assert len(imdb_boxes) == imdb.num_images, 'calculations not complete'

    # save results
    rpn_folder = os.path.join(imdb.root_path, 'rpn_data')
    if not os.path.exists(rpn_folder):
        os.mkdir(rpn_folder)

    rpn_file = os.path.join(rpn_folder, imdb.name + '_rpn.pkl')
    with open(rpn_file, 'wb') as f:
        cPickle.dump(imdb_boxes, f, cPickle.HIGHEST_PROTOCOL)

    if thresh > 0:
        full_rpn_file = os.path.join(rpn_folder, imdb.name + '_full_rpn.pkl')
        with open(full_rpn_file, 'wb') as f:
            cPickle.dump(original_boxes, f, cPickle.HIGHEST_PROTOCOL)

    logger.info('wrote rpn proposals to %s' % rpn_file)
    return imdb_boxes


def im_detect(predictor, data_batch, data_names, scale):
    output = predictor.predict(data_batch)

    data_dict = dict(zip(data_names, data_batch.data))
    if config.TEST.HAS_RPN:
        rois = output['rois_output'].asnumpy()[:, 1:]
    else:
        rois = data_dict['rois'].asnumpy().reshape((-1, 5))[:, 1:]
    im_shape = data_dict['data'].shape

    # save output
    scores = output['cls_prob_reshape_output'].asnumpy()[0]
    bbox_deltas = output['bbox_pred_reshape_output'].asnumpy()[0]

    # post processing
    pred_boxes = bbox_pred(rois, bbox_deltas)
    pred_boxes = clip_boxes(pred_boxes, im_shape[-2:])

    # we used scaled image & roi to train, so it is necessary to transform them back
    pred_boxes = pred_boxes / scale

    if config.HAS_PART:
        head_scores = output['head_prob_reshape_output'].asnumpy()[0]
        head_gids = np.argmax(head_scores, axis=1)
        head_deltas = output['head_pred_reshape_output'].asnumpy()[0]
        # means = config.TRAIN.BBOX_MEANS
        stds = np.reshape(np.array(config.TRAIN.BBOX_STDS), (-1, 4))
        head_deltas *= stds

        head_boxes = pred_head(rois, head_deltas, head_gids, config.PART_GRID_HW)
        head_boxes /= scale

        joints_scores = [output['joint_prob{}_reshape_output'.format(i)].asnumpy()[0] for i in range(4)]
        joints_gids = [np.argmax(j, axis=1) for j in joints_scores]
        joints_deltas = [output['joint_pred{}_reshape_output'.format(i)].asnumpy()[0] for i in range(4)]
        joints_deltas = [j * stds[:, :2] for j in joints_deltas]

        joints = [pred_joint(rois, jd, jid, config.PART_GRID_HW) \
                for (jd, jid) in zip(joints_deltas, joints_gids)]
        joints = np.hstack(joints)
        joints /= scale

        return scores, pred_boxes, head_boxes, joints, data_dict

    return scores, pred_boxes, data_dict


def pred_eval(predictor, test_data, imdb, vis=False, thresh=1e-3):
    """
    wrapper for calculating offline validation for faster data analysis
    in this example, all threshold are set by hand
    :param predictor: Predictor
    :param test_data: data iterator, must be non-shuffle
    :param imdb: image database
    :param vis: controls visualization
    :param thresh: valid detection threshold
    :return:
    """
    assert vis or not test_data.shuffle
    data_names = [k[0] for k in test_data.provide_data]

    nms = py_nms_wrapper(config.TEST.NMS)

    # limit detections to max_per_image over all classes
    max_per_image = -1

    num_images = imdb.num_images
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    i = 0
    t = time.time()
    for im_info, data_batch in test_data:
        t1 = time.time() - t
        t = time.time()

        scale = im_info[0, 2]
        scores, boxes, data_dict = im_detect(predictor, data_batch, data_names, scale)

        t2 = time.time() - t
        t = time.time()

        for j in range(1, imdb.num_classes):
            indexes = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[indexes, j, np.newaxis]
            cls_boxes = boxes[indexes, j * 4:(j + 1) * 4]
            cls_dets = np.hstack((cls_boxes, cls_scores))
            keep = nms(cls_dets)
            all_boxes[j][i] = cls_dets[keep, :]

        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in range(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        if vis:
            boxes_this_image = [[]] + [all_boxes[j][i] for j in range(1, imdb.num_classes)]
            vis_all_detection(data_dict['data'].asnumpy(), boxes_this_image, imdb.classes, scale)

        t3 = time.time() - t
        t = time.time()
        logger.info('testing %d/%d data %.4fs net %.4fs post %.4fs' % (i, imdb.num_images, t1, t2, t3))
        i += 1

    det_file = os.path.join(imdb.cache_path, imdb.name + '_detections.pkl')
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, protocol=cPickle.HIGHEST_PROTOCOL)

    imdb.evaluate_detections(all_boxes)


def vis_all_detection(im_array, detections, class_names, scale, head_boxes=None, joints=None):
    """
    visualize all detections in one image
    :param im_array: [b=1 c h w] in rgb
    :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
    :param class_names: list of names in imdb
    :param scale: visualize the scaled image
    :return:
    """
    import matplotlib.pyplot as plt
    import random
    im = image.transform_inverse(im_array, config.PIXEL_MEANS)
    plt.imshow(im)
    for j, name in enumerate(class_names):
        if name == '__background__':
            continue
        color = (random.random(), random.random(), random.random())  # generate a random color
        dets = detections[j]
        for det in dets:
            bbox = det[:4] * scale
            score = det[-1]
            rect = plt.Rectangle((bbox[0], bbox[1]),
                                 bbox[2] - bbox[0],
                                 bbox[3] - bbox[1], fill=False,
                                 edgecolor=color, linewidth=3.5)
            plt.gca().add_patch(rect)
            plt.gca().text(bbox[0], bbox[1] - 2,
                           '{:s} {:.3f}'.format(name, score),
                           bbox=dict(facecolor=color, alpha=0.5), fontsize=12, color='white')
    if head_boxes is not None:
        color = (random.random(), random.random(), random.random())  # generate a random color
        for det in head_boxes:
            bbox = det[:4] * scale
            rect = plt.Rectangle((bbox[0], bbox[1]),
                                 bbox[2] - bbox[0],
                                 bbox[3] - bbox[1], fill=False,
                                 edgecolor=color, linewidth=3.5)
            plt.gca().add_patch(rect)
    if joints is not None:
        color = (random.random(), random.random(), random.random())  # generate a random color
        for joint in joints:
            joint *= scale
            for (cx, cy) in zip(joint[0::2], joint[1::2]):
                ptr = plt.Circle((cx, cy), 5.0, color=color)
                plt.gca().add_artist(ptr)

    plt.show()


def draw_all_detection(im_array, detections, class_names, scale):
    """
    visualize all detections in one image
    :param im_array: [b=1 c h w] in rgb
    :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
    :param class_names: list of names in imdb
    :param scale: visualize the scaled image
    :return:
    """
    import cv2
    import random
    color_white = (255, 255, 255)
    im = image.transform_inverse(im_array, config.PIXEL_MEANS)
    # change to bgr
    im = cv2.cvtColor(im, cv2.cv.CV_RGB2BGR)
    for j, name in enumerate(class_names):
        if name == '__background__':
            continue
        color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))  # generate a random color
        dets = detections[j]
        for det in dets:
            bbox = det[:4] * scale
            score = det[-1]
            bbox = map(int, bbox)
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=color, thickness=2)
            cv2.putText(im, '%s %.3f' % (class_names[j], score), (bbox[0], bbox[1] + 10),
                        color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
    return im
