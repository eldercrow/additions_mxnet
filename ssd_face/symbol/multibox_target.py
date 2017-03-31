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
        assert self.ignore_label == -1

    def forward(self, is_train, req, in_data, out_data, aux):
        """ 
        Compute IOUs between valid labels and anchors.
        Then sample positive and negatives.
        """
        # assert len(in_data) == 2
        # inputs: ['probs_cls', 'preds_reg', 'anchors', 'label']
        # outs: ['sample_cls', 'sample_reg', 'target_cls', 'mask_cls', 'target_reg', 'mask_reg']
        probs_cls = in_data[0]
        preds_reg = in_data[1]
        labels_all = in_data[3].asnumpy() # (batch, num_label, 5)
        max_probs_cls = np.max(probs_cls.asnumpy()[:, :, 1:], axis=2)

        sample_per_batch = self.n_max_sample * (self.sample_per_label + (1 + self.hard_neg_ratio))

        n_batch, n_anchors, nch = in_data[0].shape

        # precompute some data
        if self.anchors is None:
            self.anchors = in_data[2].asnumpy()
            self.area_anchors = \
                    (self.anchors[:, 2] - self.anchors[:, 0]) * (self.anchors[:, 3] - self.anchors[:, 1])
            self.nidx_neg = [[]] * n_anchors
            self.nidx_pos = [[]] * n_anchors

        # process each batch
        sample_cls = mx.nd.zeros((n_batch, sample_per_batch, nch), ctx=in_data[0].context)
        target_cls = np.full((n_batch, sample_per_batch, nch), -1, dtype=np.float32)
        # mask_cls = np.zeros((n_batch, sample_per_batch, nch))
        sample_reg = mx.nd.zeros((n_batch, sample_per_batch, 4), ctx=in_data[0].context)
        target_reg = np.full((n_batch, sample_per_batch, 4), -1, dtype=np.float32)
        # mask_reg = np.zeros((n_batch, sample_per_batch, 4))

        anchor_locs_all = np.full((n_batch, sample_per_batch), -1, dtype=np.int32)
        for i in range(n_batch):
            pcls = probs_cls[i] # input cls
            preg = preds_reg[i] # input reg
            scls = sample_cls[i] # sampled cls
            sreg = sample_reg[i] # sampled reg
            tcls = target_cls[i] # target cls
            treg = target_reg[i] # target reg
            labels = labels_all[i, :, :]
            max_probs = max_probs_cls[i, :] # (n_anchors,)
            anchor_locs, tc, tr = self._forward_batch(labels, max_probs, nch)
            eidx = np.minimum(sample_per_batch, len(anchor_locs))
            if eidx == 0:
                continue
            for k, l in enumerate(anchor_locs[:eidx]):
                scls[k] = pcls[l]
                sreg[k] = preg[l]
            anchor_locs_all[i, :eidx] = np.array(anchor_locs)[:eidx]
            tcls[:eidx, :] = np.array(tc)[:eidx, :]
            treg[:eidx, :] = np.array(tr)[:eidx, :]

        target_cls = np.reshape(target_cls, (-1, nch))
        target_reg = np.reshape(target_reg, (-1, 4))

        self.assign(aux[0], 'write', mx.nd.array(anchor_locs_all))
        self.assign(out_data[0], req[0], mx.nd.reshape(sample_cls, (-1, nch)))
        self.assign(out_data[1], req[1], mx.nd.reshape(sample_reg, (-1, 4)))
        self.assign(out_data[2], req[2], mx.nd.array(target_cls, ctx=in_data[0].context))
        self.assign(out_data[3], req[3], mx.nd.array(target_cls[:, 0:1] != -1, ctx=in_data[0].context))
        self.assign(out_data[4], req[4], mx.nd.array(target_reg, ctx=in_data[0].context))
        self.assign(out_data[5], req[5], mx.nd.array(target_cls[:, 0:1] == 0, ctx=in_data[0].context))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        # pass the gradient to their corresponding positions
        grad_cls = mx.nd.zeros(in_grad[0].shape, ctx=in_grad[0].context) # (n n_anchor nch)
        grad_reg = mx.nd.zeros(in_grad[1].shape, ctx=in_grad[1].context) # (n n_anchor nch)
        n_batch = grad_cls.shape[0]
        target_locs = aux[0].asnumpy().astype(int)
        k = 0
        for i in range(n_batch):
            gc = grad_cls[i] # (n_anchor nch)
            gr = grad_reg[i] # (n_anchor nch)
            loc = target_locs[i, :]
            for l in loc:
                if l != -1:
                    gc[l] = out_grad[0][k]
                    gr[l] = out_grad[1][k]
                k += 1
        self.assign(in_grad[0], req[0], grad_cls)
        self.assign(in_grad[1], req[1], grad_reg)
        self.assign(in_grad[2], req[2], 0)
        self.assign(in_grad[3], req[3], 0)

    def _forward_batch(self, labels, max_probs, nch):
        """ 
        I will process each batch sequentially.
        Note that anchors are transposed, so (4, num_anchors).
        """
        n_anchors = self.anchors.shape[0]
        
        pos_anchor_locs = []
        pos_target_cls = []
        pos_target_reg = []
        neg_anchor_locs = []
        neg_target_cls = []
        neg_target_reg = []
        neg_iou = np.zeros(n_anchors)

        # valid gt labels
        labels = _get_valid_labels(labels, self.n_max_sample)
        for label in labels:
            iou = _compute_iou(label[1:], self.anchors, self.area_anchors) 
            neg_iou = np.maximum(iou, neg_iou)

            for i in range(self.sample_per_label):
                # pick positive
                midx = np.argmax(iou)
                if iou[midx] <= self.th_iou:
                    break
                if not midx in pos_anchor_locs:
                    pos_anchor_locs.append(midx)
                    target_cls = [0] * nch
                    if label[0] == -1:
                        target_cls[0] = -1
                    else:
                        target_cls[int(label[0])] = 1
                    pos_target_cls.append(target_cls)
                    target_reg = _compute_loc_target(label[1:], self.anchors[midx, :], self.variances)
                    pos_target_reg.append(target_reg.tolist())
                    # apply nms
                    if len(self.nidx_pos[midx]) == 0:
                        self.nidx_pos[midx] = _compute_nms_cands( \
                                midx, self.anchors, self.th_iou, self.area_anchors)
                    nidx = self.nidx_pos[midx]
                    iou[nidx] = 0

        # hard sample mining
        # first remove positive samples from mining
        pidx = np.where(neg_iou > self.th_neg_iou)[0]
        max_probs[pidx] = 0
        # pick hard samples one by one
        n_neg_sample = len(pos_anchor_locs) * self.hard_neg_ratio
        for i in range(n_neg_sample):
            midx = np.argmax(max_probs)
            if max_probs[midx] == 0:
                break
            if not midx in neg_anchor_locs:
                neg_anchor_locs.append(midx)
                target_cls = [0] * nch
                target_cls[0] = 1
                neg_target_cls.append(target_cls)
                neg_target_reg.append([-1] * 4)
                # appy nms
                if len(self.nidx_neg[midx]) == 0:
                    self.nidx_neg[midx] = _compute_nms_cands( \
                            midx, self.anchors, self.th_neg_iou, self.area_anchors)
                nidx = self.nidx_neg[midx]
                max_probs[nidx] = 0

        return pos_anchor_locs + neg_anchor_locs, \
                pos_target_cls + neg_target_cls, \
                pos_target_reg + neg_target_reg

def _get_valid_labels(labels, max_sample):
    n_valid_label = 0
    for label in labels:
        if all(label == -1.0): 
            break
        n_valid_label += 1
        if n_valid_label == max_sample:
            break
    return labels[:n_valid_label, :]

def _compute_nms_cands(midx, anchors, th_nms, area_anchors=None):
    #
    if area_anchors is None:
        area_anchors = (anchors[:, 2] - anchors[:, 0]) * (anchors[:, 3] - anchors[:, 1])

    iou = _compute_iou(anchors[midx], anchors, area_anchors)
    iidx = np.where(iou > th_nms)[0]
    sidx = np.argsort(iou[iidx])[::-1]
    return iidx[sidx]

def _compute_iou(label, anchors, area_anchors):
    # label: (5, )
    # anchors: (4, num_anchors)
    iw = np.maximum(0, \
            np.minimum(label[2], anchors[:, 2]) - np.maximum(label[0], anchors[:, 0]))
    ih = np.maximum(0, \
            np.minimum(label[3], anchors[:, 3]) - np.maximum(label[1], anchors[:, 1]))
    I = iw * ih
    U = (label[3] - label[1]) * (label[2] - label[0]) + area_anchors
    
    iou = I / np.maximum((U - I), 0.000001)

    return iou # (num_anchors, )

def _compute_loc_target(gt_bb, bb, variances):
    loc_target = np.zeros((4, ), dtype=np.float32)
    iw = 1.0 / (bb[2] - bb[0])
    ih = 1.0 / (bb[3] - bb[1])
    loc_target[0] = ((gt_bb[2] + gt_bb[0]) - (bb[2] + bb[0])) * 0.5 # * iw
    loc_target[1] = ((gt_bb[3] + gt_bb[1]) - (bb[3] + bb[1])) * 0.5 # * ih
    loc_target[2] = np.log2((gt_bb[2] - gt_bb[0]) * iw)
    loc_target[3] = np.log2((gt_bb[3] - gt_bb[1]) * ih)

    # if np.abs(loc_target[2]) > 15.0 or np.abs(loc_target[2]) > 15.0:
    #     import ipdb
    #     ipdb.set_trace()
    #
    # if not np.all(np.isfinite(loc_target)):
    #     import ipdb
    #     ipdb.set_trace()

    return loc_target / variances

@mx.operator.register("multibox_target")
class MultiBoxTargetProp(mx.operator.CustomOpProp):
    def __init__(self, n_class, 
            th_iou=0.5, th_neg_iou=0.35, 
            n_max_sample=128, sample_per_label=3, hard_neg_ratio=3., ignore_label=-1, 
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
        if isinstance(variances, str):
            variances = make_tuple(variances)
        self.variances = np.array(variances)

    def list_arguments(self):
        return ['probs_cls', 'preds_reg', 'anchors', 'label']

    def list_outputs(self):
        return ['sample_cls', 'sample_reg', 'target_cls', 'mask_cls', 'target_reg', 'mask_reg']

    def list_auxiliary_states(self):
        return ['target_loc_weight', ]

    def infer_shape(self, in_shape):
        n_batch = in_shape[0][0]
        n_class = in_shape[0][2]
        n_label = in_shape[3][1]
        sample_per_batch = self.n_max_sample * (self.sample_per_label + (1 + self.hard_neg_ratio))

        sample_cls_shape = [n_batch*sample_per_batch, n_class]
        sample_reg_shape = [n_batch*sample_per_batch, 4]
        target_cls_shape = sample_cls_shape
        mask_cls_shape = [n_batch*sample_per_batch, 1]
        target_reg_shape = sample_reg_shape
        mask_reg_shape = mask_cls_shape

        out_shape = [sample_cls_shape, sample_reg_shape, 
                target_cls_shape, mask_cls_shape, 
                target_reg_shape, mask_reg_shape]

        target_loc_shape = [n_batch, sample_per_batch]

        return in_shape, out_shape, [target_loc_shape, ]

    def create_operator(self, ctx, shapes, dtypes):
        return MultiBoxTarget( \
                self.n_class, self.th_iou, self.th_neg_iou, 
                self.n_max_sample, self.sample_per_label, self.hard_neg_ratio, 
                self.ignore_label, self.variances)

