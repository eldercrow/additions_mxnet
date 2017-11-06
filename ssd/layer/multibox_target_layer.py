# pylint: skip-file
import os
# MXNET_CPU_WORKER_NTHREADS must be greater than 1 for custom op to work on CPU
os.environ["MXNET_CPU_WORKER_NTHREADS"] = "8"
import mxnet as mx
import numpy as np
import logging
from ast import literal_eval as make_tuple

class MultiBoxTarget(mx.operator.CustomOp):
    """
    Python implementation of MultiBoxTarget layer.
    """
    def __init__(self, th_iou, th_iou_neg, th_nms_neg, th_small, square_bb, per_cls_reg,
            reg_sample_ratio, hard_neg_ratio, variances, ignore_labels):
        #
        super(MultiBoxTarget, self).__init__()
        self.th_iou = th_iou
        self.th_iou_neg = th_iou_neg
        self.th_nms_neg = th_nms_neg
        self.th_small = th_small
        self.square_bb = square_bb
        self.per_cls_reg = per_cls_reg
        self.reg_sample_ratio = reg_sample_ratio
        self.hard_neg_ratio = hard_neg_ratio
        self.variances = variances
        self.ignore_labels = ignore_labels

        # precompute nms candidates
        self.anchors = None
        self.nidx_neg = None
        self.anchors_t = None
        self.area_anchors_t = None

        self.th_anc_overlap = 0.6

    def forward(self, is_train, req, in_data, out_data, aux):
        """
        Compute IOUs between valid labels and anchors.
        Then sample positive and negatives.
        """
        # inputs:  ['anchors', 'label', 'probs_cls', 'preds_reg']
        # outputs: ['target_reg', 'mask_reg', 'target_cls']
        n_batch, nch, n_anchors = in_data[2].shape

        labels_all = in_data[1].asnumpy().astype(np.float32) # (batch, num_label, 6)
        labels_all = labels_all[:, :, :5]
        max_cids = mx.nd.argmax(in_data[2], axis=1).asnumpy().astype(int)
        probs_bg_cls = mx.nd.slice_axis(in_data[2], axis=1, begin=0, end=1).asnumpy()
        probs_bg_cls = 1 - np.reshape(probs_bg_cls, (n_batch, -1))

        # precompute some data
        if self.anchors_t is None:
            self.anchors = np.reshape(in_data[0].asnumpy(), (-1, 4)) # (n_anchor, 4)
            self.anchors_t = mx.nd.transpose(mx.nd.reshape(in_data[0].copy(), shape=(-1, 4)), (1, 0)) # (4, n_anchor)
            self.area_anchors_t = \
                    (self.anchors_t[2] - self.anchors_t[0]) * (self.anchors_t[3] - self.anchors_t[1])
            self.nidx_neg = [[]] * n_anchors
            # self.nidx_pos = [[]] * n_anchors
            overlaps = _compute_overlap(self.anchors_t, self.area_anchors_t, (0, 0, 1, 1))
            # self.oob_mask = (overlaps <= self.th_anc_overlap)
            self.n_class = nch
        else:
            assert self.anchors.shape == in_data[0].shape[1:]

        # numpy arrays for outputs of the layer
        target_reg = np.zeros((n_batch, n_anchors, 4), dtype=np.float32)
        mask_reg = np.zeros_like(target_reg)
        target_cls = np.full((n_batch, 1, n_anchors), -1, dtype=np.float32)
        # match_info = np.full((n_batch, in_data[1].shape[1], 50), -1)

        max_iou_pos = []

        for i in range(n_batch):
            target_cls[i][0], target_reg[i], mask_reg[i], max_iou = self._forward_batch_pos( \
                    labels_all[i], max_cids[i], target_cls[i][0], target_reg[i], mask_reg[i])
            # target_cls[i][0], target_reg[i], mask_reg[i], max_iou, match_info[i] = self._forward_batch_pos( \
            #         labels_all[i], max_cids[i], target_cls[i][0], target_reg[i], mask_reg[i])
            max_iou_pos.append(max_iou)

        # gather per batch samples
        for i in range(n_batch):
            target_cls[i][0] = self._forward_batch_neg( \
                    probs_bg_cls[i], max_iou_pos[i], target_cls[i][0])

        if self.ignore_labels:
            for i in range(n_batch):
                target_cls[i][0], mask_reg[i] = self._forward_batch_ignore( \
                        labels_all[i], target_cls[i][0], mask_reg[i])

        target_reg = np.reshape(target_reg, (n_batch, -1))
        mask_reg = np.reshape(mask_reg, (n_batch, -1))

        self.assign(out_data[0], req[0], mx.nd.array(target_reg, ctx=in_data[2].context))
        self.assign(out_data[1], req[1], mx.nd.array(mask_reg, ctx=in_data[2].context))
        self.assign(out_data[2], req[2], mx.nd.array(target_cls, ctx=in_data[2].context))
        # self.assign(out_data[3], req[3], mx.nd.array(match_info, ctx=in_data[2].context))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        '''
        Pass the gradient to their corresponding positions
        '''
        for i, r in enumerate(req):
            self.assign(in_grad[i], r, 0)

    def _forward_batch_pos(self, labels, max_cids, target_cls, target_reg, mask_reg):
        '''
        labels: (n_label, 5)
        max_cids: (n_anchor, )
        target_cls: (n_anchor, )
        target_reg: (n_anchor, 4)
        mask_reg: (n_anchor, 4)
        '''
        n_anchors = self.anchors_t.shape[1]

        # match_info = np.full((labels.shape[0], 50), -1)

        labels = _get_valid_labels(labels)
        max_iou = np.zeros(n_anchors, dtype=np.float32)

        for i, label in enumerate(labels):
            gt_cls = int(label[0]) + 1
            if self.square_bb:
                lsq = _fit_box_ratio(label[1:], 1.0)
            else:
                # lsq = label[1:]
                lsq = _autofit_ratio(label[1:])
            iou = _compute_iou(lsq, self.anchors_t, self.area_anchors_t)

            # skip already occupied ones
            iou_mask = iou > max_iou
            max_iou = np.maximum(iou, max_iou)
            if label[0] == -1:
                continue
            gt_sz = np.maximum(label[3]-label[1], label[4]-label[2])
            if gt_sz < self.th_small and np.max(iou) < self.th_iou_neg:
                continue
            # skip oob boxes
            # iou_mask = np.logical_and(iou_mask, self.oob_mask == False)

            # positive and regression samples
            pidx = np.where(np.logical_and(iou_mask, iou > self.th_iou))[0]
            if len(pidx) == 0:
                # at least one positive sample
                pidx = [np.argmax(iou)]

            target_cls[pidx] = gt_cls
            rt, rm = _compute_loc_target(label[1:], self.anchors[pidx, :], self.variances)
            if self.per_cls_reg:
                rt, rm = _expand_target(rt, gt_cls, self.n_class)
            target_reg[pidx, :] = rt
            mask_reg[pidx, :] = rm

            # ridx = np.where(np.logical_and(iou_mask, iou > self.th_iou_neg))[0]
            # ridx = np.setdiff1d(ridx, pidx)
            # ridx = ridx[max_cids[ridx] == gt_cls]
            # ridx = ridx[target_cls[ridx] == -1]
            # n_reg_sample = len(pidx) * self.reg_sample_ratio
            # if len(ridx) > n_reg_sample:
            #     ridx = np.random.choice(ridx, n_reg_sample, replace=False)
            # target_cls[ridx] = -1
            # rt, rm = _compute_loc_target(label[1:], self.anchors[ridx, :], self.variances)
            # target_reg[ridx, :] = rt
            # mask_reg[ridx, :] = rm

            # if len(pidx) > 50:
            #     pidx = pidx[:50]
            # match_info[i, :len(pidx)] = pidx

        return target_cls, target_reg, mask_reg, max_iou #, match_info

    def _forward_batch_neg(self, bg_probs, max_iou, target_cls):
        '''
        '''
        # mask for negative sampling region
        rmask = np.logical_or(max_iou > self.th_iou_neg, target_cls > 0)

        # no hard negative sampling case
        if self.hard_neg_ratio <= 0:
            target_cls[rmask == False] = 0
            return target_cls

        # first remove positive samples from mining
        ridx = np.where(rmask)[0]
        bg_probs[ridx] = -1.0

        # number of hard samples
        n_neg_sample = int(np.sum(target_cls > 0) * self.hard_neg_ratio)
        # if n_neg_sample == 0:
        #     logging.info("No negative sample, will put one at least.")
        n_neg_sample = np.maximum(n_neg_sample, 1)

        neg_probs = []
        neg_anchor_locs = []

        eidx = np.argsort(bg_probs)[::-1]
        nidx = []

        # pick hard samples one by one, with nms
        k = 0
        for ii in eidx:
            # if bg_probs[ii] < 0.0 or self.oob_mask[ii]:
            #     continue
            #
            target_cls[ii] = 0
            nidx.append(ii)
            if self.th_nms_neg < 1.0:
                # apply nms
                if len(self.nidx_neg[ii]) == 0:
                    self.nidx_neg[ii] = _compute_nms_cands( \
                            self.anchors[ii], self.anchors_t, self.area_anchors_t, self.th_nms_neg)
                idx = self.nidx_neg[ii]
                bg_probs[idx] = -1
            k += 1
            if k >= n_neg_sample:
                break

        return target_cls

    def _forward_batch_ignore(self, labels, target_cls, mask_reg):
        #
        for l in labels:
            if not l[0] in self.ignore_labels:
                continue
            # compute overlaps between anchors and the label
            overlaps = _compute_overlap(self.anchors_t, self.area_anchors_t, l[1:])
            iidx = np.where(overlaps > self.th_anc_overlap)[0]
            target_cls[iidx] = -1
            mask_reg[iidx, :] = 0
        return target_cls, mask_reg


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
    sf = 192.0 / (anc[3] - anc[1])
    if sf > 1.0:
        anc = _rescale_anchor(anc, sf)
        anc_t = _rescale_anchor(anchors_t, sf)
        aanc_t = (anc_t[3]-anc_t[1]) * (anc_t[2]-anc_t[0])
        iou = _compute_iou(anc, anc_t, aanc_t)
    else:
        iou = _compute_iou(anc, anchors_t, area_anchors_t)
    iidx = np.where(iou > th_nms)[0]
    return iidx

def _compute_iou(label, anchors_t, area_anchors_t):
    #
    iw = mx.nd.minimum(label[2], anchors_t[2]) - mx.nd.maximum(label[0], anchors_t[0])
    ih = mx.nd.minimum(label[3], anchors_t[3]) - mx.nd.maximum(label[1], anchors_t[1])
    I = mx.nd.maximum(iw, 0) * mx.nd.maximum(ih, 0)
    U = (label[3] - label[1]) * (label[2] - label[0]) + area_anchors_t

    iou = I / mx.nd.maximum((U - I), 1e-08)
    return iou.asnumpy() # (num_anchors, )

def _compute_overlap(anchors_t, area_anchors_t, img_shape):
    #
    iw = mx.nd.minimum(img_shape[2], anchors_t[2]) - mx.nd.maximum(img_shape[0], anchors_t[0])
    ih = mx.nd.minimum(img_shape[3], anchors_t[3]) - mx.nd.maximum(img_shape[1], anchors_t[1])
    I = mx.nd.maximum(iw, 0) * mx.nd.maximum(ih, 0)
    overlap = I / mx.nd.maximum(area_anchors_t, 1e-08)
    return overlap.asnumpy()

def _compute_loc_target(gt_bb, bb, variances):
    eps = 1e-07
    loc_target = np.zeros_like(bb)
    aw = (bb[:, 2] - bb[:, 0])
    ah = (bb[:, 3] - bb[:, 1])
    loc_target[:, 0] = ((gt_bb[2] + gt_bb[0]) - (bb[:, 2] + bb[:, 0])) * 0.5 / aw
    loc_target[:, 1] = ((gt_bb[3] + gt_bb[1]) - (bb[:, 3] + bb[:, 1])) * 0.5 / ah
    loc_target[:, 2] = np.log(np.maximum((gt_bb[2] - gt_bb[0]) / aw, np.finfo(np.float32).eps))
    loc_target[:, 3] = np.log(np.maximum((gt_bb[3] - gt_bb[1]) / ah, np.finfo(np.float32).eps))
    return loc_target / variances, np.ones_like(loc_target)

def _rescale_anchor(anchors_t, sf):
    ranc = anchors_t.copy()
    ranc[0] = (ranc[0] + ranc[2]) * 0.5
    ranc[1] = (ranc[1] + ranc[3]) * 0.5
    ranc[2] = ranc[0] + (anchors_t[2] - anchors_t[0]) * 0.5 * sf
    ranc[3] = ranc[1] + (anchors_t[3] - anchors_t[1]) * 0.5 * sf
    ranc[0] -= (anchors_t[2] - anchors_t[0]) * 0.5 * sf
    ranc[1] -= (anchors_t[3] - anchors_t[1]) * 0.5 * sf
    return ranc

def _fit_box_ratio(bb, ratio):
    #
    ndim = bb.ndim
    if ndim == 1:
        bb = np.reshape(bb, (1, -1))

    cx = (bb[:, 0] + bb[:, 2]) / 2.0
    cy = (bb[:, 1] + bb[:, 3]) / 2.0
    sz2 = np.maximum(bb[:, 2] - bb[:, 0], bb[:, 3] - bb[:, 1]) / 2.0
    res = bb.copy()
    res[:, 0] = cx - sz2 * ratio
    res[:, 1] = cy - sz2
    res[:, 2] = cx + sz2 * ratio
    res[:, 3] = cy + sz2
    if ndim == 1:
        res = res.ravel()
    return res


def _autofit_ratio(bb):
    #
    ww = bb[2] - bb[0]
    hh = bb[3] - bb[1]
    cx = (bb[0] + bb[2]) / 2.0
    cy = (bb[1] + bb[3]) / 2.0

    ratio = ww / hh
    if ratio > 2.0:
        hh = ww * 0.5
    elif ratio < 0.5:
        ww = hh * 0.5

    res = bb.copy()
    res[0] = cx - ww * 0.5
    res[1] = cy - hh * 0.5
    res[2] = cx + ww * 0.5
    res[3] = cy + hh * 0.5
    return res


def _expand_target(loc_target, cid, n_cls):
    n_target = loc_target.shape[0]
    loc_target_e = np.zeros((n_target, 4 * n_cls), dtype=np.float32)
    loc_mask_e = np.zeros_like(loc_target_e)
    sidx = cid * 4
    loc_target_e[:, sidx:sidx+4] = loc_target
    loc_mask_e[:, sidx:sidx+4] = 1
    # for i in range(n_target):
    #     loc_target_e[i, sidx:sidx+4] = loc_target
    #     loc_mask_e[i, sidx:sidx+4] = 1
    return loc_target_e, loc_mask_e


@mx.operator.register("multibox_target")
class MultiBoxTargetProp(mx.operator.CustomOpProp):
    def __init__(self,
            th_iou=0.5, th_iou_neg=0.4, th_nms_neg=1.0,
            th_small=0.04, square_bb=False, per_cls_reg=False,
            reg_sample_ratio=1.0, hard_neg_ratio=3.0,
            variances=(0.1, 0.1, 0.2, 0.2), ignore_labels=''):
        #
        super(MultiBoxTargetProp, self).__init__(need_top_grad=False)
        self.th_iou = float(th_iou)
        self.th_iou_neg = float(th_iou_neg)
        self.th_nms_neg = float(th_nms_neg)
        self.th_small = float(th_small)
        self.reg_sample_ratio = int(reg_sample_ratio)
        self.hard_neg_ratio = float(hard_neg_ratio)
        self.per_cls_reg = bool(make_tuple(str(per_cls_reg)))
        self.square_bb = bool(make_tuple(str(square_bb)))
        if isinstance(variances, str):
            variances = make_tuple(variances)
        self.variances = np.reshape(np.array(variances), (1, -1))
        try:
            self.ignore_labels = make_tuple(ignore_labels)
            if isinstance(self.ignore_labels, int):
                self.ignore_labels = [self.ignore_labels,]
        except:
            self.ignore_labels = []

    def list_arguments(self):
        return ['anchors', 'label', 'probs_cls']

    def list_outputs(self):
        return ['target_reg', 'mask_reg', 'target_cls'] #, 'match_info']

    def infer_shape(self, in_shape):
        n_batch, n_class, n_sample = in_shape[2]

        target_reg_shape = (n_batch, n_sample*4)
        if self.per_cls_reg:
            target_reg_shape[1] *= n_class
        mask_reg_shape = target_reg_shape
        target_cls_shape = (n_batch, 1, n_sample)
        # match_info_shape = (n_batch, in_shape[1][1], 50)

        out_shape = [target_reg_shape, mask_reg_shape, target_cls_shape] #, match_info_shape]
        return in_shape, out_shape, []

    def create_operator(self, ctx, shapes, dtypes):
        return MultiBoxTarget( \
                self.th_iou, self.th_iou_neg, self.th_nms_neg,
                self.th_small, self.square_bb, self.per_cls_reg,
                self.reg_sample_ratio, self.hard_neg_ratio,
                self.variances, self.ignore_labels)
