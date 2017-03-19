# pylint: skip-file
import os
# MXNET_CPU_WORKER_NTHREADS must be greater than 1 for custom op to work on CPU
os.environ["MXNET_CPU_WORKER_NTHREADS"] = "2"
import mxnet as mx
import numpy as np
import logging

class MultiBoxTarget(mx.operator.CustomOp):
    """ 
    Python implementation of MultiBoxTarget layer. 
    """
    def __init__(self, n_class, th_iou, th_neg_iou, hard_neg_ratio, ignore_label, variances):
        super(MultiBoxTarget, self).__init__()
        self.n_class = int(n_class)
        self.th_iou = float(th_iou)
        self.th_neg_iou = float(th_neg_iou)
        self.hard_neg_ratio = float(hard_neg_ratio)
        self.ignore_label = float(ignore_label)
        self.variances = variances.asnumpy()

    def forward(self, is_train, req, in_data, out_data, aux):
        """ 
        Compute IOUs between valid labels and anchors.
        Then sample positive and negatives.
        """
        # assert len(in_data) == 2
        anchors = in_data[0].asnumpy() # (batch, num_anchor, 4)
        labels_all = in_data[1].asnumpy() # (batch, num_label, 5)

        n_batch = labels_all.shape[0]
        n_anchors = anchors.shape[1] 
        cls_targets = np.zeros((n_batch, n_anchors), dtype=np.float32)
        loc_targets = np.zeros((n_batch, n_anchors, 4), dtype=np.float32)
        loc_masks = np.zeros((n_batch, n_anchors), dtype=np.float32)
        # compute IOUs
        for i in range(n_batch):
            labels = labels_all[i, :, :]
            cls_target, loc_target, loc_mask = self._forward_batch(anchors[0, :, :], labels)
            cls_targets[i, :] = cls_target
            loc_targets[i, :, :] = loc_target
            loc_masks[i, :] = loc_mask

        loc_masks = np.tile(loc_masks[:, :, np.newaxis], (1, 1, 4))
        cls_targets = np.reshape(cls_targets, (n_batch, 1, n_anchors))
        
        self.assign(out_data[0], req[0], mx.nd.array(loc_targets, ctx=in_data[0].context))
        self.assign(out_data[1], req[1], mx.nd.array(loc_masks, ctx=in_data[1].context))
        self.assign(out_data[2], req[2], mx.nd.array(cls_targets, ctx=in_data[0].context))


    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        pass

    def _forward_batch(self, anchors, labels):
        """ 
        I will process each batch sequentially.
        Note that anchors are transposed, so (4, num_anchors).
        """
        n_anchors = anchors.shape[0]
        anchors_t = np.transpose(anchors, (1, 0))
        # outputs: cls_target, loc_target, loc_mask
        cls_target = self.ignore_label * np.ones((n_anchors,), dtype=np.float32)
        loc_target = np.zeros((n_anchors, 4), dtype=np.float32)
        loc_mask = np.zeros((n_anchors,), dtype=np.float32)

        # valid gt labels
        n_valid_label = 0
        for label in labels:
            if label[0] == -1.0: 
                break
            n_valid_label += 1

        if self.hard_neg_ratio == 0:
            cls_target *= 0

        neg_iou = np.zeros((n_anchors,), dtype=np.float32)
        n_valid_pos = 0
        for i in range(n_valid_label):
            label = labels[i, :]
            # cls_inx = int(label[0].asscalar())
            iou = _compute_IOU(label, anchors_t) 
            # pick positive
            midx = np.argmax(iou)
            if iou[midx] > self.th_iou:
                cls_target[midx] = label[0] + 1 # since 0 is the background
                loc_target[midx] = _compute_loc_target(label[1:], anchors[midx, :], self.variances)
                loc_mask[midx] = 1
                iou[midx] = 1
                n_valid_pos += 1
            neg_iou = np.maximum(iou, neg_iou)
        # assert n_valid_pos > 0, "No valid positive samples!"
        # pick negatives, if hard sample mining needed
        if self.hard_neg_ratio > 0 and n_valid_pos > 0:
            neg_iou *= neg_iou < self.th_neg_iou
            neg_iou += np.random.uniform(0.0, 0.1, size=neg_iou.shape).astype(np.float32)
            n_neg_sample = np.minimum(n_anchors - n_valid_pos, n_valid_pos * self.hard_neg_ratio)
            sidx = np.argsort(neg_iou)[::-1]
            nidx = sidx[int(n_neg_sample)-1]
            cls_target[neg_iou >= neg_iou[nidx]] = 0
        return cls_target, loc_target, loc_mask

def _compute_IOU(label, anchors):
    # label: (5, )
    # anchors: (4, num_anchors)
    iw = np.maximum(0, \
            np.minimum(label[3], anchors[2, :]) - np.maximum(label[1], anchors[0, :]))
    ih = np.maximum(0, \
            np.minimum(label[4], anchors[3, :]) - np.maximum(label[2], anchors[1, :]))
    I = iw * ih
    U = (label[4] - label[2]) * (label[3] - label[1]) + \
            (anchors[2, :] - anchors[0, :]) * (anchors[3, :] - anchors[1, :])
    
    iou = I / np.maximum((U - I), 0.000001)

    return iou # (num_anchors, )

def _compute_loc_target(gt_bb, bb, variances):
    loc_target = np.zeros((4, ), dtype=np.float32)
    loc_target[0] = ((gt_bb[2]+gt_bb[0]) / 2.0 - (bb[2]-bb[0]) / 2.0) / variances[0]
    loc_target[1] = ((gt_bb[3]+gt_bb[1]) / 2.0 - (bb[3]-bb[1]) / 2.0) / variances[1]
    loc_target[2] = np.log2((gt_bb[2]-gt_bb[0]) / (bb[2]-bb[0])) / variances[2]
    loc_target[3] = np.log2((gt_bb[3]-gt_bb[1]) / (bb[3]-bb[1])) / variances[3]

    if np.abs(loc_target[2]) > 15.0 or np.abs(loc_target[2]) > 15.0:
        import ipdb
        ipdb.set_trace()

    if not np.all(np.isfinite(loc_target)):
        import ipdb
        ipdb.set_trace()

    return loc_target

@mx.operator.register("multibox_target")
class MultiBoxTargetProp(mx.operator.CustomOpProp):
    def __init__(self, n_class, 
            th_iou=0.5, th_neg_iou=0.35, hard_neg_ratio=1., ignore_label=-1, variances=(0.1, 0.1, 0.2, 0.2)):
        super(MultiBoxTargetProp, self).__init__(need_top_grad=False)
        self.n_class = int(n_class)
        self.th_iou = float(th_iou)
        self.th_neg_iou = float(th_neg_iou)
        self.hard_neg_ratio = float(hard_neg_ratio)
        self.ignore_label = float(ignore_label)
        self.variances = mx.nd.array(np.array(variances).astype(float))

    def list_arguments(self):
        return ['anchors', 'label', 'cls_preds']

    def list_outputs(self):
        return ['cls_target', 'loc_target', 'loc_mask']

    def infer_shape(self, in_shape):
        n_batch = in_shape[1][0] 
        n_anchors = in_shape[0][1] 

        cls_target_shape = (n_batch, 1, n_anchors)
        loc_target_shape = (n_batch, n_anchors, 4)
        loc_mask_shape = (n_batch, n_anchors, 4)

        return [in_shape[0], in_shape[1], in_shape[2]], \
                [loc_target_shape, loc_mask_shape, cls_target_shape], \
                []

    def create_operator(self, ctx, shapes, dtypes):
        return MultiBoxTarget( \
                self.n_class, self.th_iou, self.th_neg_iou, self.hard_neg_ratio, self.ignore_label, self.variances)

