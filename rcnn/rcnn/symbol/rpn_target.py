# pylint: skip-file
import os
# MXNET_CPU_WORKER_NTHREADS must be greater than 1 for custom op to work on CPU
os.environ["MXNET_CPU_WORKER_NTHREADS"] = "4"
import mxnet as mx
import numpy as np
import logging
from ast import literal_eval as make_tuple

from ..processing.bbox_transform import bbox_overlaps

class RPNTarget(mx.operator.CustomOp):
    """ 
    Python implementation of MultiBoxTarget layer. 
    """
    def __init__(self, th_iou, rpn_batch_size, pos_ratio, ignore_label):
        #
        super(RPNTarget, self).__init__(need_top_grad=False)
        self.th_iou = th_iou
        self.rpn_batch_size = rpn_batch_size
        self.pos_ratio = pos_ratio
        self.ignore_label = ignore_label
        assert self.ignore_label == -1


    def forward(self, is_train, req, in_data, out_data, aux):
        """ 
        Compute IOUs between valid labels and anchors.
        Then sample positive and negatives.
        """
        # inputs: ['anchors', 'gt_boxes']
        # outs: ['rpn_labels', 'bbox_targets', 'bbox_weights']
        n_batch, _, n_anchor, _ = in_data[0].shape
        anchors = in_data[0].asnumpy().astype(np.float32)
        anchors = np.reshape(np.transpose(anchors, (0, 2, 1, 3)), (-1, 4))
        oob_mask = in_data[1].asnumpy().astype(bool).ravel()
        gt_boxes_all = in_data[2].asnumpy().astype(np.float32) # (batch, num_label, 5)

        rpn_labels = np.zeros((n_batch, 1, n_anchor, 1))
        bbox_targets = np.zeros((n_batch, 4, n_anchor, 1))
        bbox_weights = np.zeros((n_batch, 4, n_anchor, 1))

        for i in range(n_batch):
            rpn_labels[i], bbox_targets[i], bbox_weights[i] = \
                    self._forward_batch(anchors, oob_mask, gt_boxes_all[i])

        out_ctx = in_data[0].context
        for k, out_array in enumerate((rpn_labels, bbox_targets, bbox_weights)):
            self.assign(out_data[k], req[k], mx.nd.array(out_array, ctx=out_ctx))


    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        # pass the gradient to their corresponding positions
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)


    def _forward_batch(self, anchors, oob_mask, gt_boxes):
        # handle each batch
        n_anchor = anchors.shape[0]

        labels = np.full((n_anchor,), -1, dtype=np.float32)

        # valid gt labels
        gt_boxes = _get_valid_labels(gt_boxes)
        if gt_boxes.size > 0:
            overlaps = bbox_overlaps(anchors.astype(np.float), gt_boxes.astype(np.float))
            argmax_overlaps = overlaps.argmax(axis=1)
            max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
            gt_argmax_overlaps = overlaps.argmax(axis=0)
            gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
            gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

            # negative
            labels[max_overlaps < self.th_iou] = 0

            # positive
            labels[gt_argmax_overlaps] = 1
            labels[max_overlaps >= self.th_iou] = 1
        else:
            labels[:] = 0

        # ignore oob anchors
        labels[oob_mask] = -1

        # subsample positive labels if we have too many
        num_fg = self.pos_ratio * self.rpn_batch_size
        fg_inds = np.where(labels == 1)[0]
        if len(fg_inds) > num_fg:
            disable_inds = npr.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            # if DEBUG:
            #     disable_inds = fg_inds[:(len(fg_inds) - num_fg)]
            labels[disable_inds] = -1

        # subsample negative labels if we have too many
        num_bg = self.rpn_batch_size - np.sum(labels == 1)
        bg_inds = np.where(labels == 0)[0]
        if len(bg_inds) > num_bg:
            disable_inds = npr.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
            # if DEBUG:
            #     disable_inds = bg_inds[:(len(bg_inds) - num_bg)]
            labels[disable_inds] = -1

        bbox_targets = np.zeros((n_anchor, 4), dtype=np.float32)
        if gt_boxes.size > 0:
            bbox_targets[:] = bbox_transform(anchors, gt_boxes[argmax_overlaps, :4])

        bbox_weights = np.zeros((n_anchor, 4), dtype=np.float32)
        bbox_weights[labels==1, :] = np.array((1, 1, 1, 1))

        labels = np.transpose(np.reshape(labels, (n_anchor, 1, 1)), (1, 0, 2))
        bbox_targets = np.transpose(np.reshape(bbox_targets, (n_anchor, 4, 1)), (1, 0, 2))
        bbox_weights = np.transpose(np.reshape(bbox_targets, (n_anchor, 4, 1)), (1, 0, 2))

        return labels, bbox_targets, bbox_weights


def _get_valid_labels(labels):
    #
    n_valid_label = 0
    for label in labels:
        if np.all(label == -1.0): 
            break
        n_valid_label += 1
        # if n_valid_label == max_sample:
        #     break
    return labels[:n_valid_label, :]


@mx.operator.register("rpn_target")
class RPNTargetProp(mx.operator.CustomOpProp):
    def __init__(self, th_iou=0.5, rpn_batch_size=256, pos_ratio=0.5, ignore_label=-1):
        #
        super(RPNTargetProp, self).__init__(need_top_grad=True)
        self.th_iou = float(th_iou)
        self.rpn_batch_size = int(rpn_batch_size)
        self.pos_ratio = float(pos_ratio)
        self.ignore_label = float(ignore_label)

    def list_arguments(self):
        return ['anchors', 'oob_mask', 'gt_boxes']

    def list_outputs(self):
        return ['rpn_labels', 'bbox_targets', 'bbox_weights']

    def infer_shape(self, in_shape):
        # anchors: (1, 4, n_anchor, 1)
        # oob_mask: (1, 1, n_anchor, 1)
        # gt_boxes: (1, n_gt, 5)
        assert in_shape[0][0] == 1, 'Only single batch is accepted.'
        assert in_shape[0][1] == 4
        assert in_shape[2][2] == 5 

        n_batch = in_shape[0][0]
        n_anchor = in_shape[0][2]
        out_shape = [(n_batch, 1, n_anchor, 1), (n_batch, 4, n_anchor, 1), (n_batch, 4, n_anchor, 1)]
        return in_shape, out_shape, []

    def create_operator(self, ctx, shapes, dtypes):
        return RPNTarget(self.th_iou, self.rpn_batch_size, self.pos_ratio, self.ignore_label)

