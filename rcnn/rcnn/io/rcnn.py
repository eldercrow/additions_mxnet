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

"""
Fast R-CNN:
data =
    {'data': [num_images, c, h, w],
    'rois': [num_rois, 5]}
label =
    {'label': [num_rois],
    'bbox_target': [num_rois, 4 * num_classes],
    'bbox_weight': [num_rois, 4 * num_classes]}
roidb extended format [image_index]
    ['image', 'height', 'width', 'flipped',
     'boxes', 'gt_classes', 'gt_overlaps', 'max_classes', 'max_overlaps', 'bbox_targets']
"""

import numpy as np
import numpy.random as npr

from ..config import config
from ..io.image import get_image, tensor_vstack
from ..processing.bbox_transform import bbox_overlaps, bbox_transform
from ..processing.bbox_regression import expand_bbox_regression_targets
from ..processing.part_transform import transform_head, transform_joint


def get_rcnn_testbatch(roidb):
    """
    return a dict of testbatch
    :param roidb: ['image', 'flipped'] + ['boxes']
    :return: data, label, im_info
    """
    assert len(roidb) == 1, 'Single batch only'
    imgs, roidb = get_image(roidb)
    im_array = imgs[0]
    im_info = np.array([roidb[0]['im_info']], dtype=np.float32)

    im_rois = roidb[0]['boxes']
    rois = im_rois
    batch_index = 0 * np.ones((rois.shape[0], 1))
    rois_array = np.hstack((batch_index, rois))[np.newaxis, :]

    data = {'data': im_array,
            'rois': rois_array}
    label = {}

    return data, label, im_info


def get_rcnn_batch(roidb):
    """
    return a dict of multiple images
    :param roidb: a list of dict, whose length controls batch size
    ['images', 'flipped'] + ['gt_boxes', 'boxes', 'gt_overlap'] => ['bbox_targets']
    :return: data, label
    """
    num_images = len(roidb)
    imgs, roidb = get_image(roidb)
    im_array = tensor_vstack(imgs)

    assert config.TRAIN.BATCH_ROIS % config.TRAIN.BATCH_IMAGES == 0, \
        'BATCHIMAGES {} must divide BATCH_ROIS {}'.format(config.TRAIN.BATCH_IMAGES, config.TRAIN.BATCH_ROIS)
    rois_per_image = config.TRAIN.BATCH_ROIS / config.TRAIN.BATCH_IMAGES
    fg_rois_per_image = np.round(config.TRAIN.FG_FRACTION * rois_per_image).astype(int)

    rois_array = list()
    labels_array = list()
    bbox_targets_array = list()
    bbox_weights_array = list()

    for im_i in range(num_images):
        roi_rec = roidb[im_i]

        im_rois, labels, bbox_targets, bbox_weights = \
            sample_rois(roi_rec, fg_rois_per_image, rois_per_image)

        # project im_rois
        # do not round roi
        rois = im_rois
        batch_index = im_i * np.ones((rois.shape[0], 1))
        rois_array_this_image = np.hstack((batch_index, rois))
        rois_array.append(rois_array_this_image)

        # add labels
        labels_array.append(labels)
        bbox_targets_array.append(bbox_targets)
        bbox_weights_array.append(bbox_weights)

    rois_array = np.array(rois_array)
    labels_array = np.array(labels_array)
    bbox_targets_array = np.array(bbox_targets_array)
    bbox_weights_array = np.array(bbox_weights_array)

    data = {'data': im_array,
            'rois': rois_array}
    label = {'label': labels_array,
             'bbox_target': bbox_targets_array,
             'bbox_weight': bbox_weights_array}

    return data, label


def sample_rois(rois, fg_rois_per_image, rois_per_image, num_classes, gt_boxes,
                gt_head_boxes=None, gt_joints=None):
    """
    generate random sample of ROIs comprising foreground and background examples
    :param rois: all_rois [n, 4]; e2e: [n, 5] with batch_index
    :param fg_rois_per_image: foreground roi number
    :param rois_per_image: total roi number
    :param num_classes: number of classes
    :param labels: maybe precomputed
    :param overlaps: maybe precomputed (max_overlaps)
    :param bbox_targets: maybe precomputed
    :param gt_boxes: optional for e2e [n, 5] (x1, y1, x2, y2, cls)
    :return: (labels, rois, bbox_targets, bbox_weights)
    """
    # infer num_classes from gt_overlaps
    # num_classes = roi_rec['gt_overlaps'].shape[1]

    # label = class RoI has max overlap with
    # rois = roi_rec['boxes']
    # labels = roi_rec['max_classes']
    # overlaps = roi_rec['max_overlaps']
    # bbox_targets = roi_rec['bbox_targets']

    overlaps = bbox_overlaps(rois[:, 1:].astype(np.float), gt_boxes[:, :4].astype(np.float))
    gt_assignment = overlaps.argmax(axis=1)
    overlaps = overlaps.max(axis=1)
    labels = gt_boxes[gt_assignment, 4]

    # foreground RoI with FG_THRESH overlap
    fg_indexes = np.where(overlaps >= config.TRAIN.FG_THRESH)[0]
    # guard against the case when an image has fewer than fg_rois_per_image foreground RoIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_indexes.size)
    # Sample foreground regions without replacement
    if len(fg_indexes) > fg_rois_per_this_image:
        fg_indexes = npr.choice(fg_indexes, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_indexes = np.where((overlaps < config.TRAIN.BG_THRESH_HI) & (overlaps >= config.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image, bg_indexes.size)
    # Sample foreground regions without replacement
    if len(bg_indexes) > bg_rois_per_this_image:
        bg_indexes = npr.choice(bg_indexes, size=bg_rois_per_this_image, replace=False)

    # indexes selected
    keep_indexes = np.append(fg_indexes, bg_indexes)
    neg_idx = np.where(overlaps < config.TRAIN.FG_THRESH)[0]
    neg_rois = rois[neg_idx]
    # pad more to ensure a fixed minibatch size
    while keep_indexes.shape[0] < rois_per_image:
        gap = np.minimum(len(neg_rois), rois_per_image - keep_indexes.shape[0])
        gap_indexes = npr.choice(range(len(neg_rois)), size=gap, replace=False)
        keep_indexes = np.append(keep_indexes, neg_idx[gap_indexes])

    # select labels
    labels = labels[keep_indexes]
    # set labels of bg_rois to be 0
    labels[fg_rois_per_this_image:] = 0
    rois = rois[keep_indexes]

    # compute bbox_target
    targets = bbox_transform(rois[:, 1:], gt_boxes[gt_assignment[keep_indexes], :4])
    if config.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED:
        targets = ((targets - np.array(config.TRAIN.BBOX_MEANS))
                   / np.array(config.TRAIN.BBOX_STDS))
    bbox_target_data = np.hstack((labels[:, np.newaxis], targets))

    bbox_targets, bbox_weights = \
        expand_bbox_regression_targets(bbox_target_data, num_classes)

    # part handling
    if config.HAS_PART and gt_head_boxes is not None:
        gids_head, targets_head, weights_head, gids_joint, targets_joint, weights_joint = \
                _sample_part_info(rois, fg_indexes, gt_head_boxes, gt_joints, gt_assignment)
        return (rois, labels, bbox_targets, bbox_weights, \
                gids_head, targets_head, weights_head, gids_joint, targets_joint, weights_joint)
    else:
        return rois, labels, bbox_targets, bbox_weights


def _sample_part_info(rois, fg_indexes, head_bbs, joints, gt_assignment):
    '''
    Required part information (currently MPII only):
        * Grid id for head location.
        * Head regression target w.r.t grid info.
        * Grid id for each joint location.
        * Joint regression target w.r.t grid info.
    '''
    n_roi = rois.shape[0]
    n_fg = len(fg_indexes)

    gids_head = np.full((n_roi,), -1, dtype=np.float32)
    targets_head = np.zeros((n_roi, 4), dtype=np.float32)
    weights_head = np.zeros_like(targets_head)

    gids_joint = np.full((n_roi, 4), -1, dtype=np.float32)
    targets_joint = np.zeros((n_roi, 8), dtype=np.float32)
    weights_joint = np.zeros_like(targets_joint)

    gt_inds = gt_assignment[fg_indexes]
    rois = rois[:n_fg]
    head_bbs = head_bbs[gt_inds]
    joints = joints[gt_inds]

    gids_head[:n_fg], targets_head[:n_fg], weights_head[:n_fg] = \
            transform_head(rois[:, 1:], head_bbs, config.PART_GRID_HW)
    # gids_head[:n_fg] += 1

    part_gids = {}
    part_targets = {}
    part_weights = {}
    joint_names = ('lshoulder', 'rshoulder', 'lhip', 'rhip')
    for i, j in enumerate(joint_names):
        part_gids[j], part_targets[j], part_weights[j] = \
                transform_joint(rois[:, 1:], joints[:, (i*3):(i+1)*3], config.PART_GRID_HW)

    gids_joint[:n_fg] = np.hstack([part_gids[j] for j in joint_names])
    # gids_joint[:n_fg] += 1
    targets_joint[:n_fg] = np.hstack([part_targets[j] for j in joint_names])
    weights_joint[:n_fg] = np.hstack([part_weights[j] for j in joint_names])

    if config.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED:
        bbox_means = np.reshape(np.array(config.TRAIN.BBOX_MEANS), (-1, 4))
        bbox_stds = np.reshape(np.array(config.TRAIN.BBOX_STDS), (-1, 4))
        targets_head[:n_fg] = (targets_head[:n_fg] - bbox_means) / bbox_stds
        targets_joint[:n_fg] = (targets_joint[:n_fg] - np.tile(bbox_means[:, :2], (1, 4))) / \
                np.tile(bbox_stds[:, :2], (1, 4))

    return gids_head, targets_head, weights_head, gids_joint, targets_joint, weights_joint
