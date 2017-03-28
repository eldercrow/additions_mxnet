# pylint: skip-file
import os
# MXNET_CPU_WORKER_NTHREADS must be greater than 1 for custom op to work on CPU
os.environ["MXNET_CPU_WORKER_NTHREADS"] = "2"
import mxnet as mx
import numpy as np
import logging
from ast import literal_eval as make_tuple

class MultiBoxTarget(mx.operator.CustomOp):
    """ 
    Python implementation of MultiBoxTarget layer. 
    """
    def __init__(self, n_class, th_iou, th_neg_iou, 
            n_max_sample, sample_per_label, hard_neg_ratio, 
            ignore_label, variances):
        #
        super(MultiBoxTarget, self).__init__()
        self.n_class = n_class
        self.th_iou = th_iou
        self.th_neg_iou = th_neg_iou
        self.n_max_sample = n_max_sample
        self.sample_per_label = sample_per_label
        self.hard_neg_ratio = hard_neg_ratio
        self.ignore_label = ignore_label
        self.variances = variances
        # precompute nms candidates
        self.nidx_neg = None
        self.nidx_pos = None
        self.anchors = None
        self.area_anchors = None

    def forward(self, is_train, req, in_data, out_data, aux):
        """ 
        Compute IOUs between valid labels and anchors.
        Then sample positive and negatives.
        """
        # assert len(in_data) == 2
        pred_conv = in_data[0]
        labels_all = in_data[2].asnumpy() # (batch, num_label, 5)
        max_probs_all = in_data[3].asnumpy() # (batch, num_anchor, num_class)

        sample_per_batch = n_max_sample * (self.sample_per_label + (1 + self.hard_neg_ratio))

        # precompute some data
        if self.anchors is None:
            self.anchors = in_data[1].asnumpy()
            self.area_anchors = (anchors[:, 2] - anchors[:, 0]) * (anchors[:, 3] - anchors[:, 1])
            self.nidx_neg = _compute_nms_cands(anchors, self.th_neg_iou, self.area_anchors)
            self.nidx_pos = _compute_nms_cands(anchors, self.th_iou, self.area_anchors)
        
        n_batch, n_anchors, nch = in_data[0].shape

        # process each batch
        target_conv = mx.nd.zeros((n_batch, sample_per_batch, nch), ctx=in_data[0].context)
        anchor_locs_all = np.full((n_batch, sample_per_batch), -1, dtype=np.uint32)
        targets_all = np.full((n_batch, sample_per_batch, 5), -1, dtype=np.float32)
        for i in range(n_batch):
            pconv = pred_conv[i]
            tconv = target_conv[i]
            labels = labels_all[i, :, :]
            max_probs = max_probs_all[i, :, :]
            anchor_locs, targets = self._forward_batch(labels, max_probs)
            eidx = np.minimum(sample_per_batch, len(anchor_locs))

            for k, l in enumerate(anchor_locs[:eidx]):
                tconv[k] = pconv[l]
            anchor_locs_all[i, :eidx] = np.array(anchor_locs)[:eidx]
            targets_all[i, :eidx, :] = np.array(targets)[:eidx, :]

        targets_all = np.reshape(targets_all, (-1, 5))

        self.assign(aux[0], 'write', mx.nd.array(anchor_locs_all))
        self.assign(out_data[0], req[0], mx.nd.reshape(target_conv, (-1, nch)))
        self.assign(out_data[1], req[1], mx.nd.array(targets_all, ctx=in_data[0].context))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        # pass the gradient to their corresponding positions
        grad_all = mx.nd.zeros(in_grad[0].shape, ctx=in_grad[0].context) # (n n_anchor nch)
        n_batch = grad_all.shape[0]
        target_locs = aux[0].asnumpy().astype(int)
        k = 0
        for i in range(n_batch):
            gradf = grad_all[i] # (n_anchor nch)
            loc = target_locs[i, :]
            for l in loc:
                if l != -1:
                    gradf[l] = out_grad[0][k]
                k += 1
        self.assign(in_grad[0], req[0], grad_all)
        self.assign(in_grad[1], req[1], 0)
        self.assign(in_grad[2], req[2], 0)
        self.assign(in_grad[3], req[3], 0)

    def _forward_batch(self, labels, max_probs):
        """ 
        I will process each batch sequentially.
        Note that anchors are transposed, so (4, num_anchors).
        """
        n_anchors = self.anchors.shape[0]
        
        pos_anchor_locs = []
        pos_targets = []
        neg_anchor_locs = []
        neg_targets = []
        neg_iou = np.zeros(n_anchors)

        # valid gt labels
        labels = _get_valid_labels(labels, self.n_max_sample)
        for label in labels:
            iou = _compute_iou(label, self.anchors, self.area_anchors) 
            neg_iou = np.maximum(iou, neg_iou)

            for i in range(self.sample_per_label):
                # pick positive
                midx = np.argmax(iou)
                if iou[midx] <= self.th_iou:
                    break
                if not midx in pos_anchor_locs:
                    pos_anchor_locs.append(midx)
                    target = np.hstack((label[0], 
                        _compute_loc_target(label[1:], self.anchors[midx, :], self.variances)))
                    pos_targets.append(target)
                    # apply nms
                    nidx = self.nidx_pos[midx]
                    iou[nidx] = 0

        # hard sample mining
        # first remove positive samples from mining
        pidx = neg_iou > th_neg_iou
        max_probs[pidx] = 0
        # pick hard samples one by one
        n_neg_sample = len(pos_anchor_locs) * self.hard_neg_ratio
        for i in range(n_neg_sample):
            midx = np.argmax(max_probs)
            if max_probs[midx] == 0:
                break
            if not midx in neg_anchor_locs:
                neg_anchor_locs.append(midx)
                neg_targets.append(np.zeros(5))
                # appy nms
                nidx = self.nidx_neg[midx]
                max_probs[nidx] = 0

        return pos_anchor_locs + neg_anchor_locs, pos_targets + neg_targets

def _get_valid_labels(labels, max_sample):
    n_valid_label = 0
    for label in labels:
        if label[0] == -1.0: 
            break
        n_valid_label += 1
        if n_valid_label == max_sample:
            break
    return labels[:n_valid_label, :]

def _compute_nms_cands(anchors, th_nms, area_anchors=None):
    #
    if area_anchors is None:
        area_anchors = (anchors[:, 2] - anchors[:, 0]) * (anchors[:, 3] - anchors[:, 1])

    nms_cands = []
    for i, anc in enumerate(anchors):
        iou = _compute_iou(anc, anchors, area_anchors)
        sidx = np.argsort(iou)[::-1]
        iou = iou[sidx]
        nidx = iou > th_nms
        nms_cands.append(sidx[nidx])
    return nms_cands

def _compute_iou(label, anchors, area_anchors):
    # label: (5, )
    # anchors: (4, num_anchors)
    iw = np.maximum(0, \
            np.minimum(label[2], anchors[2, :]) - np.maximum(label[0], anchors[0, :]))
    ih = np.maximum(0, \
            np.minimum(label[3], anchors[3, :]) - np.maximum(label[1], anchors[1, :]))
    I = iw * ih
    U = (label[3] - label[1]) * (label[2] - label[0]) + area_anchors
    
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
            th_iou=0.5, th_neg_iou=0.35, 
            n_max_sample=64, sample_per_label=3, hard_neg_ratio=1., ignore_label=-1, 
            variances=(0.1, 0.1, 0.2, 0.2)):
        #
        super(MultiBoxTargetProp, self).__init__(need_top_grad=True)
        self.n_class = int(n_class)
        self.th_iou = float(th_iou)
        self.th_neg_iou = float(th_neg_iou)
        self.n_max_sample = int(n_max_sample)
        self.sample_per_label = int(sample_per_label)
        self.hard_neg_ratio = int(hard_neg_ratio)
        self.ignore_label = float(ignore_label)
        self.variances = np.array(make_tuple(variances))

    def list_arguments(self):
        return ['pred_conv', 'anchors', 'label', 'cls_probs']

    def list_outputs(self):
        return ['pred_target', 'target_cls', 'target_reg', 'mask_reg']

    def infer_shape(self, in_shape):
        n_batch = in_shape[0][0]
        nch = in_shape[0][2]
        n_label = in_shape[2][1]
        sample_per_batch = n_label * (self.sample_per_label + (1 + self.hard_neg_ratio))

        pred_target_shape = (n_batch*sample_per_batch, nch)
        target_cls_shape = (n_batch*sample_per_batch, )
        target_reg_shape = (n_batch*sample_per_batch, 4)
        mask_reg_shape = (n_batch*sample_per_batch, 1)

        target_loc_shape = (n_batch, sample_per_batch)

        return in_shape, \
                [pred_target_shape, target_cls_shape, target_reg_shape, mask_reg_shape], \
                [target_loc_shape, ]

    def create_operator(self, ctx, shapes, dtypes):
        return MultiBoxTarget( \
                self.n_class, self.th_iou, self.th_neg_iou, 
                self.n_max_sample, self.sample_per_label, self.hard_neg_ratio, 
                self.ignore_label, self.variances)

