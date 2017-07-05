# pylint: skip-file
import os
# MXNET_CPU_WORKER_NTHREADS must be greater than 1 for custom op to work on CPU
os.environ["MXNET_CPU_WORKER_NTHREADS"] = "4"
import mxnet as mx
import numpy as np
import logging
from ast import literal_eval as make_tuple

class MultiBoxTarget2(mx.operator.CustomOp):
    """
    Python implementation of MultiBoxTarget layer.
    """
    def __init__(self, n_class, img_wh, th_iou, th_iou_neg, th_nms_neg, max_pos_sample,
            reg_sample_ratio, hard_neg_ratio, ignore_label, variances, per_cls_reg):
        #
        super(MultiBoxTarget2, self).__init__()
        self.n_class = n_class
        self.img_shape = (0, 0, img_wh[0], img_wh[1])
        self.th_iou = th_iou
        self.th_iou_neg = th_iou_neg
        self.th_nms_neg = th_nms_neg
        self.max_pos_sample = max_pos_sample
        self.reg_sample_ratio = reg_sample_ratio
        self.hard_neg_ratio = hard_neg_ratio
        self.ignore_label = ignore_label
        self.variances = variances
        self.per_cls_reg = per_cls_reg

        # precompute nms candidates
        self.nidx_neg = None
        self.anchors_t = None
        self.area_anchors_t = None

        self.th_anc_overlap = 0.6
        self.max_sample = int(self.max_pos_sample + \
                          self.reg_sample_ratio * self.max_pos_sample + \
                          self.hard_neg_ratio * self.max_pos_sample)

    def forward(self, is_train, req, in_data, out_data, aux):
        """
        Compute IOUs between valid labels and anchors.
        Then sample positive and negatives.
        """
        # assert len(in_data) == 2
        # inputs: ['preds_cls', 'preds_reg', 'anchors', 'label']
        # outs: ['sample_cls', 'sample_reg', 'target_cls', 'target_reg', 'mask_reg']
        n_batch, n_anchors, nch = in_data[0].shape

        preds_cls = in_data[0] # (n_batch, n_anchor, nch)
        preds_reg = in_data[1]
        labels_all = in_data[3].asnumpy().astype(np.float32) # (batch, num_label, 5)
        probs_cls = mx.nd.slice_axis(in_data[4], axis=1, begin=1, end=None)
        max_probs_cls = mx.nd.max(probs_cls, axis=1).asnumpy()
        max_cids = mx.nd.argmax(probs_cls, axis=1).asnumpy().astype(int) + 1
        probs_cls = in_data[4].asnumpy()

        # precompute some data
        if self.anchors_t is None:
            self.anchors = in_data[2].asnumpy()
            self.anchors_t = mx.nd.transpose(in_data[2], (1, 0)).copy()
            self.area_anchors_t = \
                    (self.anchors_t[2] - self.anchors_t[0]) * (self.anchors_t[3] - self.anchors_t[1])
            self.nidx_neg = [[]] * n_anchors
            # self.nidx_pos = [[]] * n_anchors
            overlaps = _compute_overlap(self.anchors_t, self.area_anchors_t, self.img_shape)
            self.oob_mask = (overlaps <= self.th_anc_overlap)
        else:
            assert self.anchors.shape == in_data[2].shape

        # process each batch
        sample_cls = mx.nd.zeros((self.max_sample, nch), ctx=in_data[0].context)
        target_cls = np.full((self.max_sample), -1, dtype=np.float32)

        nch_reg = 4 * self.n_class if self.per_cls_reg else 4
        sample_reg = mx.nd.zeros((self.max_sample, nch_reg), ctx=in_data[0].context)
        target_reg = np.full((self.max_sample, nch_reg), -1, dtype=np.float32)
        mask_reg = np.zeros((self.max_sample, nch_reg), dtype=np.float32)

        anchor_locs_all = np.full((self.max_sample, 2), -1, dtype=np.int32)

        # positive samples
        anchor_locs_pos = np.empty((0, 2), dtype=int)
        tc_pos = np.empty((0,), dtype=np.int32)
        tr_pos = np.empty((0, 4))
        mr_pos = np.empty((0, 4))
        # probs_pos = []
        max_iou_pos = []
        n_pos_sample = 0
        for i in range(n_batch):
            anchor_locs, tc, tr, mr, max_iou = \
                    self._forward_batch_pos(labels_all[i], probs_cls[i], max_cids[i], i)
            anchor_locs_pos = np.vstack((anchor_locs_pos, anchor_locs))
            tc_pos = np.append(tc_pos, tc)
            tr_pos = np.vstack((tr_pos, tr))
            mr_pos = np.vstack((mr_pos, mr))
            # probs_pos += p
            max_iou_pos.append(max_iou)
            n_pos_sample = np.maximum(n_pos_sample, np.sum(np.array(tc) > 0))

        # negative samples
        n_neg_sample = np.maximum(1, np.round(self.hard_neg_ratio * n_pos_sample))
        anchor_locs_neg = []
        probs_neg = []
        # gather per batch samples
        for i in range(n_batch):
            anchor_locs, p = self._forward_batch_neg(n_neg_sample, max_probs_cls[i], max_iou_pos[i], i)
            anchor_locs_neg += anchor_locs
            probs_neg += p

        # hard negative sampling with whole batchs
        n_max_sample_neg = np.sum(np.array(tc_pos) > 0) * self.hard_neg_ratio
        sidx = np.argsort(np.array(probs_neg))[::-1]
        if len(sidx) > n_max_sample_neg:
            sidx = sidx[:n_max_sample_neg]
        anchor_locs_neg = np.array(anchor_locs_neg)[sidx, :]

        # subsample pos/reg/neg samples if we have too many
        max_sample = int(1 + self.reg_sample_ratio) * self.max_pos_sample
        if len(tc_pos) > max_sample:
            pidx = np.random.choice(np.arange(len(tc_pos)), max_sample, replace=False)
            anchor_locs_pos = anchor_locs_pos[pidx]
            tc_pos = tc_pos[pidx]
            tr_pos = tr_pos[pidx, :]
            mr_pos = mr_pos[pidx, :]

        max_sample = int(self.hard_neg_ratio) * self.max_pos_sample
        if len(anchor_locs_neg) > max_sample:
            pidx = np.random.choice(np.arange(len(anchor_locs_neg)), max_sample, replace=False)
            anchor_locs_neg = anchor_locs_neg[pidx]

        # gather and arrange samples we will use

        k = 0
        for i, aloc in enumerate(anchor_locs_pos):
            if k >= self.max_sample:
                break
            bid = aloc[0]
            aid = aloc[1]

            target_cls[k] = tc_pos[i]
            sample_cls[k] = preds_cls[bid][aid]
            target_reg[k] = tr_pos[i]
            mask_reg[k] = mr_pos[i]
            sample_reg[k] = preds_reg[bid][aid]
            anchor_locs_all[k] = aloc
            k += 1

        for aloc in anchor_locs_neg:
            if k >= self.max_sample:
                break
            bid = aloc[0]
            aid = aloc[1]

            target_cls[k] = 0
            sample_cls[k] = preds_cls[bid][aid]
            sample_reg[k] = preds_reg[bid][aid]
            anchor_locs_all[k] = aloc
            k += 1

        # mask_reg = np.reshape(mask_reg, (-1, 1))

        self.assign(aux[0], 'write', mx.nd.array(anchor_locs_all))
        self.assign(out_data[0], req[0], sample_cls)
        self.assign(out_data[1], req[1], sample_reg)
        self.assign(out_data[2], req[2], mx.nd.array(target_cls, ctx=in_data[0].context))
        self.assign(out_data[3], req[3], mx.nd.array(target_reg, ctx=in_data[0].context))
        self.assign(out_data[4], req[4], mx.nd.array(mask_reg, ctx=in_data[0].context))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        # pass the gradient to their corresponding positions
        grad_cls = mx.nd.zeros(in_grad[0].shape, ctx=in_grad[0].context) # (n n_anchor nch)
        grad_reg = mx.nd.zeros(in_grad[1].shape, ctx=in_grad[1].context) # (n n_anchor nch)
        n_batch = grad_cls.shape[0]
        target_locs = aux[0].asnumpy().astype(int)
        for i, tloc in enumerate(target_locs):
            bid = tloc[0]
            aid = tloc[1]
            if bid == -1 or aid == -1:
                continue
            grad_cls[bid][aid] = out_grad[0][i]
            grad_reg[bid][aid] = out_grad[1][i]

        self.assign(in_grad[0], req[0], grad_cls)
        self.assign(in_grad[1], req[1], grad_reg)
        self.assign(in_grad[2], req[2], 0)
        self.assign(in_grad[3], req[3], 0)
        self.assign(in_grad[4], req[4], 0)

    def _forward_batch_pos(self, labels, probs, max_cids, batch_id):
        n_anchors = self.anchors_t.shape[1]

        cls_target = np.zeros((n_anchors,), dtype=np.int32)
        reg_target = np.zeros((n_anchors, 4))
        reg_mask = np.zeros((n_anchors, 4))

        labels = _get_valid_labels(labels)
        max_iou = np.zeros(n_anchors)

        for label in labels:
            gt_cls = int(label[0])
            iou = _compute_iou(label[1:], self.anchors_t, self.area_anchors_t)

            # skip already occupied ones
            iou_mask = iou > max_iou
            max_iou = np.maximum(iou, max_iou)
            if label[0] == -1:
                continue
            # skip oob boxes
            iou_mask = np.logical_and(iou_mask, self.oob_mask == False)

            # positive samples
            pidx = np.where(np.logical_and(iou_mask, iou > self.th_iou))[0]
            cls_target[pidx] = gt_cls
            rt, rm = _compute_loc_target(label[1:], self.anchors[pidx, :], self.variances)
            if self.per_cls_reg:
                rt, rm = _expand_target(rt, gt_cls, self.n_class)
            reg_target[pidx, :] = rt
            reg_mask[pidx, :] = rm

            # regression samples
            # ridx = np.where(np.logical_and(iou_mask, iou > self.th_iou_neg))[0]
            # ridx = np.setdiff1d(ridx, pidx)
            # if not self.per_cls_reg:
            #     ridx = ridx[max_cids[ridx] == gt_cls]
            # if ridx.size > pidx.size * self.reg_sample_ratio:
            #     ridx = np.random.choice(ridx, pidx.size * self.reg_sample_ratio, replace=False)
            # cls_target[ridx] = -1
            # rt, rm = _compute_loc_target(label[1:], self.anchors[ridx, :], self.variances)
            # if self.per_cls_reg:
            #     rt, rm = _expand_target(rt, gt_cls, self.n_class)
            # reg_target[ridx, :] = rt
            # reg_mask[ridx, :] = rm

        # gather data
        anc_idx = np.where(cls_target != 0)[0]
        cls_target = cls_target[anc_idx]
        reg_target = reg_target[anc_idx, :]
        reg_mask = reg_mask[anc_idx, :]
        anchor_locs = np.zeros((len(anc_idx), 2), dtype=np.int32)
        anchor_locs[:, 0] = batch_id
        anchor_locs[:, 1] = anc_idx
        return anchor_locs, cls_target, reg_target, reg_mask, max_iou

    def _forward_batch_neg(self, n_neg_sample, max_probs, neg_iou, batch_id):
        # first remove positive samples from mining
        pidx = np.where(neg_iou > 1.0/3.0)[0]
        max_probs[pidx] = -1.0

        neg_probs = []
        neg_anchor_locs = []

        eidx = np.argsort(max_probs)[::-1]

        # pick hard samples one by one, with nms
        for ii in eidx:
            if max_probs[ii] < 0.0 or self.oob_mask[ii]:
                continue

            neg_probs.append(max_probs[ii])
            neg_anchor_locs.append((batch_id, ii))
            # apply nms
            if len(self.nidx_neg[ii]) == 0:
                self.nidx_neg[ii] = _compute_nms_cands( \
                        self.anchors[ii], self.anchors_t, self.area_anchors_t, self.th_nms_neg)
            nidx = self.nidx_neg[ii]
            # nidx = ii
            max_probs[nidx] = -1
            if len(neg_anchor_locs) >= n_neg_sample:
                break

        return neg_anchor_locs, neg_probs

def _get_valid_labels(labels):
    #
    n_valid_label = 0
    for label in labels:
        if np.all(label == -1.0):
            break
        n_valid_label += 1
    return labels[:n_valid_label, :]

def _compute_nms_cands(anc, anchors_t, area_anchors_t, th_nms):
    #
    iou = _compute_iou(anc, anchors_t, area_anchors_t)
    iidx = np.where(iou > th_nms)[0]
    return iidx

def _compute_iou(label, anchors_t, area_anchors_t):
    #
    iw = mx.nd.minimum(label[2], anchors_t[2]) - mx.nd.maximum(label[0], anchors_t[0])
    ih = mx.nd.minimum(label[3], anchors_t[3]) - mx.nd.maximum(label[1], anchors_t[1])
    I = mx.nd.maximum(iw, 0) * mx.nd.maximum(ih, 0)
    U = (label[3] - label[1]) * (label[2] - label[0]) + area_anchors_t

    iou = I / mx.nd.maximum((U - I), 0.000001)
    return iou.asnumpy() # (num_anchors, )

def _compute_overlap(anchors_t, area_anchors_t, img_shape):
    #
    iw = mx.nd.minimum(img_shape[2], anchors_t[2]) - mx.nd.maximum(img_shape[0], anchors_t[0])
    ih = mx.nd.minimum(img_shape[3], anchors_t[3]) - mx.nd.maximum(img_shape[1], anchors_t[1])
    I = mx.nd.maximum(iw, 0) * mx.nd.maximum(ih, 0)
    overlap = I / area_anchors_t
    return overlap.asnumpy()

def _compute_loc_target(gt_bb, bb, variances):
    loc_target = np.zeros_like(bb)
    aw = (bb[:, 2] - bb[:, 0])
    ah = (bb[:, 3] - bb[:, 1])
    loc_target[:, 0] = ((gt_bb[2] + gt_bb[0]) - (bb[:, 2] + bb[:, 0])) * 0.5 / aw
    loc_target[:, 1] = ((gt_bb[3] + gt_bb[1]) - (bb[:, 3] + bb[:, 1])) * 0.5 / ah
    loc_target[:, 2] = np.log2((gt_bb[2] - gt_bb[0]) / aw)
    loc_target[:, 3] = np.log2((gt_bb[3] - gt_bb[1]) / ah)
    return loc_target / variances, np.ones_like(loc_target)

def _expand_target(loc_target, cid, n_cls):
    n_target = loc_target.shape[0]
    loc_target_e = np.zeros((n_target, 4 * n_cls), dtype=np.float32)
    loc_mask_e = np.zeros_like(loc_target_e)
    for i, c in enumerate(cid):
        sidx = c * 4
        loc_target_e[i, sidx:sidx+4] = loc_target
        loc_mask_e[i, sidx:sidx+4] = 1
    return loc_target_e, loc_mask_e


@mx.operator.register("multibox_target2")
class MultiBoxTargetProp2(mx.operator.CustomOpProp):
    def __init__(self, n_class, img_wh,
            th_iou=0.5, th_iou_neg=1.0/3.0, th_nms_neg=1.0/2.0, max_pos_sample=512,
            reg_sample_ratio=2.0, hard_neg_ratio=3.0, ignore_label=-1,
            variances=(0.1, 0.1, 0.2, 0.2), per_cls_reg=False):
        #
        super(MultiBoxTargetProp2, self).__init__(need_top_grad=True)
        self.n_class = int(n_class)
        if isinstance(img_wh, str):
            img_wh = make_tuple(img_wh)
        self.img_wh = img_wh
        self.th_iou = float(th_iou)
        self.th_iou_neg = float(th_iou_neg)
        self.th_nms_neg = float(th_nms_neg)
        self.max_pos_sample = int(max_pos_sample)
        self.reg_sample_ratio = int(reg_sample_ratio)
        self.hard_neg_ratio = int(hard_neg_ratio)
        self.ignore_label = float(ignore_label)
        assert self.ignore_label == -1
        if isinstance(variances, str):
            variances = make_tuple(variances)
        self.variances = np.reshape(np.array(variances), (1, -1))
        self.per_cls_reg = bool(make_tuple(str(per_cls_reg)))

        self.max_sample = int(self.max_pos_sample + \
                          self.reg_sample_ratio * self.max_pos_sample + \
                          self.hard_neg_ratio * self.max_pos_sample)

    def list_arguments(self):
        return ['preds_cls', 'preds_reg', 'anchors', 'label', 'probs_cls']

    def list_outputs(self):
        return ['sample_cls', 'sample_reg', 'target_cls', 'target_reg', 'mask_reg']

    def list_auxiliary_states(self):
        return ['target_loc_weight']

    def infer_shape(self, in_shape):
        n_class = in_shape[0][2]

        sample_cls_shape = (self.max_sample, n_class)
        if self.per_cls_reg:
            sample_reg_shape = (self.max_sample, 4 * n_class)
        else:
            sample_reg_shape = (self.max_sample, 4)
        target_cls_shape = (self.max_sample, )
        target_reg_shape = sample_reg_shape
        mask_reg_shape = sample_reg_shape

        out_shape = [sample_cls_shape, sample_reg_shape,
                target_cls_shape, target_reg_shape, mask_reg_shape]

        target_loc_shape = (self.max_sample, 2)
        return in_shape, out_shape, [target_loc_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return MultiBoxTarget2( \
                self.n_class, self.img_wh, self.th_iou, self.th_iou_neg, self.th_nms_neg,
                self.max_pos_sample, self.reg_sample_ratio, self.hard_neg_ratio,
                self.ignore_label, self.variances, self.per_cls_reg)
