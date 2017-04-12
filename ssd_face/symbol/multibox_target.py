# pylint: skip-file
import os
# MXNET_CPU_WORKER_NTHREADS must be greater than 1 for custom op to work on CPU
os.environ["MXNET_CPU_WORKER_NTHREADS"] = "4"
import mxnet as mx
import numpy as np
import logging
from ast import literal_eval as make_tuple

class MultiBoxTarget(mx.operator.CustomOp):
    """ 
    Python implementation of MultiBoxTarget layer. 
    """
    def __init__(self, n_class, th_iou, th_nms, th_neg_nms, 
            n_max_label, sample_per_label, hard_neg_ratio, 
            ignore_label, variances):
        #
        super(MultiBoxTarget, self).__init__()
        self.n_class = n_class
        self.th_iou = th_iou
        self.th_nms = th_nms
        self.th_neg_nms = th_neg_nms
        self.n_max_label = n_max_label
        self.sample_per_label = sample_per_label
        self.hard_neg_ratio = hard_neg_ratio
        self.ignore_label = ignore_label
        self.variances = variances
        # precompute nms candidates
        self.nidx_neg = None
        self.nidx_pos = None
        self.anchors_t = None
        self.area_anchors_t = None
        assert self.ignore_label == -1

        self.mean_pos_prob = 0.5

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
        self.anchors = in_data[2]
        labels_all = in_data[3].asnumpy().astype(np.float32) # (batch, num_label, 5)
        # softmax
        probs_cls = mx.nd.transpose(preds_cls, axes=(2, 0, 1)) # (nch, n_batch, n_anchor)
        probs_cls = mx.nd.broadcast_sub(probs_cls, probs_cls[0])
        probs_cls = mx.nd.exp(probs_cls[1:])
        probs_cls = mx.nd.broadcast_div(probs_cls, mx.nd.sum(probs_cls, axis=0, keepdims=True) + 1.0)
        max_probs_cls = mx.nd.zeros((1, n_batch, n_anchors), ctx=probs_cls.context)
        for i in range(nch-1):
            max_probs_cls = mx.nd.maximum(max_probs_cls, probs_cls[i])
        max_probs_cls = mx.nd.transpose(max_probs_cls, (1, 2, 0)).asnumpy() # (n_batch, n_anchor, 1)
        max_probs_cls[np.isnan(max_probs_cls)] = 0
        max_probs_cls = np.reshape(max_probs_cls, (n_batch, n_anchors))

        sample_per_batch = n_batch * self.n_max_label * (self.sample_per_label + (1 + self.hard_neg_ratio))

        # precompute some data
        # self.anchors = in_data[2].asnumpy()
        # self.area_anchors = \
        #         (self.anchors[:, 2] - self.anchors[:, 0]) * (self.anchors[:, 3] - self.anchors[:, 1])
        if self.anchors_t is None:
            self.anchors_t = mx.nd.transpose(in_data[2], (1, 0)).copy()
            self.area_anchors_t = \
                    (self.anchors_t[2] - self.anchors_t[0]) * (self.anchors_t[3] - self.anchors_t[1])
            self.nidx_neg = [[]] * n_anchors
            self.nidx_pos = [[]] * n_anchors

        # process each batch
        sample_cls = mx.nd.zeros((sample_per_batch, nch), ctx=in_data[0].context)
        target_cls = np.full((sample_per_batch), -1, dtype=np.float32)
        sample_reg = mx.nd.zeros((sample_per_batch, 4), ctx=in_data[0].context)
        target_reg = np.full((sample_per_batch, 4), -1, dtype=np.float32)
        anchor_locs_all = np.full((sample_per_batch, 2), -1, dtype=np.int32)

        # positive samples
        anchor_locs_pos = []
        tc_pos = []
        tr_pos = []
        probs_pos = []
        max_iou_pos = []
        n_pos_sample = 0
        for i in range(n_batch):
            anchor_locs, tc, tr, p, max_iou = self._forward_batch_pos(labels_all[i], max_probs_cls[i], i)
            anchor_locs_pos += anchor_locs
            tc_pos += tc
            tr_pos += tr
            probs_pos += p
            max_iou_pos.append(max_iou)
            n_pos_sample = np.maximum(n_pos_sample, len(anchor_locs))

        if len(probs_pos) > 0:
            mean_pos_prob = np.mean(np.array(probs_pos))
            a = 1e-04 * len(probs_pos)
            self.mean_pos_prob = mean_pos_prob * a + (1.0-a) * self.mean_pos_prob

        # negative samples
        n_neg_sample = 2 * np.maximum(1, self.hard_neg_ratio * n_pos_sample)
        anchor_locs_neg = []
        probs_neg = []
        for i in range(n_batch):
            anchor_locs, p = self._forward_batch_neg(n_neg_sample, max_probs_cls[i], max_iou_pos[i], i)
            anchor_locs_neg += anchor_locs
            probs_neg += p

        n_max_sample_neg = len(anchor_locs_pos) * self.hard_neg_ratio

        # sidx = np.random.permutation(np.arange(len(probs_neg)))
        sidx = np.argsort(np.array(probs_neg))[::-1]
        if len(sidx) > n_max_sample_neg:
            sidx = sidx[:n_max_sample_neg]
        anchor_locs_neg = np.array(anchor_locs_neg)[sidx, :]

        k = 0
        for i, aloc in enumerate(anchor_locs_pos):
            if k >= sample_per_batch:
                break
            bid = aloc[0]
            aid = aloc[1]

            target_cls[k] = tc_pos[i]
            sample_cls[k] = preds_cls[bid][aid]
            target_reg[k] = tr_pos[i]
            sample_reg[k] = preds_reg[bid][aid]
            anchor_locs_all[k] = aloc
            k += 1

        for aloc in anchor_locs_neg:
            if k >= sample_per_batch:
                break
            bid = aloc[0]
            aid = aloc[1]

            target_cls[k] = 0
            sample_cls[k] = preds_cls[bid][aid]
            sample_reg[k] = preds_reg[bid][aid]
            anchor_locs_all[k] = aloc
            k += 1

        mask_reg = np.reshape(target_cls > 0, (-1, 1))

        self.assign(aux[0], 'write', mx.nd.array(anchor_locs_all))
        self.assign(aux[1], 'write', mx.nd.array([self.mean_pos_prob]))
        self.assign(out_data[0], req[0], mx.nd.reshape(sample_cls, (-1, nch)))
        self.assign(out_data[1], req[1], mx.nd.reshape(sample_reg, (-1, 4)))
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

    def _forward_batch_pos(self, labels, max_probs, batch_id):
        n_anchors = self.anchors_t.shape[1]
        
        pos_anchor_locs = []
        pos_target_cls = []
        pos_target_reg = []
        pos_probs = []
        max_iou = np.zeros(n_anchors)
        is_sampled = np.zeros(n_anchors)

        # valid gt labels
        labels = _get_valid_labels(labels)
        np.random.shuffle(labels)
        for label in labels:
            iou = _compute_iou(label[1:], self.anchors_t, self.area_anchors_t) 
            max_iou = np.maximum(iou, max_iou)
            if label[0] == -1:
                continue

            pidx = np.where(iou > self.th_iou)[0]
            if pidx.size == 0:
                pidx = np.array([np.argmax(iou)])
            pos_iou = iou[pidx]
            sidx = np.argsort(pos_iou)[::-1]
            pidx = pidx[sidx]

            k = 0
            for ii in pidx:
                # pick positive
                if iou[ii] <= self.th_iou:
                    continue
                # iou[ii] = 0
                if is_sampled[ii] < iou[ii]:
                    if is_sampled[ii] > 0:
                        didx = pos_anchor_locs.index((batch_id, ii))
                        del pos_anchor_locs[didx]
                        del pos_target_cls[didx]
                        del pos_target_reg[didx]
                        del pos_probs[didx]
                        k -= 1
                    is_sampled[ii] = iou[ii]
                    pos_anchor_locs.append((batch_id, ii))
                    pos_target_cls.append(label[0])
                    target_reg = _compute_loc_target(label[1:], self.anchors[ii].asnumpy(), self.variances)
                    pos_target_reg.append(target_reg.tolist())
                    pos_probs.append(max_probs[ii])
                    # # apply nms
                    # if len(self.nidx_pos[ii]) == 0:
                    #     self.nidx_pos[ii] = _compute_nms_cands( \
                    #             self.anchors[ii], self.anchors_t, self.area_anchors_t, self.th_nms)
                    # # assert len(self.nidx_pos) > 0
                    # nidx = self.nidx_pos[ii]
                    # iou[nidx] = 0.0
                    k += 1
                    if k >= self.sample_per_label:
                        break

        n_max_sample = self.n_max_label * self.sample_per_label

        if len(pos_anchor_locs) >= n_max_sample:
            pos_anchor_locs = pos_anchor_locs[:n_max_sample]
            pos_target_cls = pos_target_cls[:n_max_sample]
            pos_target_reg = pos_target_reg[:n_max_sample]
        return pos_anchor_locs, pos_target_cls, pos_target_reg, pos_probs, max_iou

    def _forward_batch_neg(self, n_neg_sample, max_probs, neg_iou, batch_id):
        # first remove positive samples from mining
        pidx = np.where(neg_iou > 1.0/3.0)[0]
        max_probs[pidx] = -1.0

        neg_probs = []
        neg_anchor_locs = []

        eidx = np.argsort(max_probs)[::-1]

        # semi-hard sample
        # hidx = np.where(max_probs > self.mean_pos_prob)[0]
        # np.random.shuffle(hidx)
        # eidx[:len(hidx)] = hidx

        # pick hard samples one by one
        for ii in eidx:
            if max_probs[ii] < 0.0:
                continue

            neg_probs.append(max_probs[ii])
            neg_anchor_locs.append((batch_id, ii))
            # apply nms
            if len(self.nidx_neg[ii]) == 0:
                self.nidx_neg[ii] = _compute_nms_cands( \
                        self.anchors[ii], self.anchors_t, self.area_anchors_t, self.th_neg_nms)
            # assert len(self.nidx_neg) > 0
            nidx = self.nidx_neg[ii]
            max_probs[nidx] = -1
            if len(neg_anchor_locs) >= n_neg_sample:
                break

        return neg_anchor_locs, neg_probs

def _get_valid_labels(labels):
    n_valid_label = 0
    for label in labels:
        if np.all(label == -1.0): 
            break
        n_valid_label += 1
        # if n_valid_label == max_sample:
        #     break
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

def _compute_loc_target(gt_bb, bb, variances):
    loc_target = np.zeros((4, ), dtype=np.float32)
    aw = (bb[2] - bb[0])
    ah = (bb[3] - bb[1])
    loc_target[0] = ((gt_bb[2] + gt_bb[0]) - (bb[2] + bb[0])) * 0.5 / aw 
    loc_target[1] = ((gt_bb[3] + gt_bb[1]) - (bb[3] + bb[1])) * 0.5 / ah 
    loc_target[2] = np.log2((gt_bb[2] - gt_bb[0]) / aw)
    loc_target[3] = np.log2((gt_bb[3] - gt_bb[1]) / ah)

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
            th_iou=0.5, th_nms=0.65, th_neg_nms=1.0/5.0, 
            n_max_label=768, sample_per_label=5, hard_neg_ratio=3., ignore_label=-1, 
            variances=(0.1, 0.1, 0.2, 0.2)):
        #
        super(MultiBoxTargetProp, self).__init__(need_top_grad=True)
        self.n_class = int(n_class)
        self.th_iou = float(th_iou)
        self.th_nms = float(th_nms)
        self.th_neg_nms = float(th_neg_nms)
        self.n_max_label = int(n_max_label)
        self.sample_per_label = int(sample_per_label)
        self.hard_neg_ratio = int(hard_neg_ratio)
        self.ignore_label = float(ignore_label)
        if isinstance(variances, str):
            variances = make_tuple(variances)
        self.variances = np.array(variances)

    def list_arguments(self):
        return ['probs_cls', 'preds_reg', 'anchors', 'label']

    def list_outputs(self):
        return ['sample_cls', 'sample_reg', 'target_cls', 'target_reg', 'mask_reg']

    def list_auxiliary_states(self):
        return ['target_loc_weight', 'mean_pos_prob_bias']

    def infer_shape(self, in_shape):
        n_batch = in_shape[0][0]
        n_class = in_shape[0][2]
        n_label = in_shape[3][1]
        sample_per_batch = self.n_max_label * (self.sample_per_label + (1 + self.hard_neg_ratio))

        sample_cls_shape = (n_batch*sample_per_batch, n_class)
        sample_reg_shape = (n_batch*sample_per_batch, 4)
        target_cls_shape = (n_batch*sample_per_batch, )
        target_reg_shape = sample_reg_shape
        mask_reg_shape = (n_batch*sample_per_batch, 1)

        out_shape = [sample_cls_shape, sample_reg_shape, 
                target_cls_shape, target_reg_shape, mask_reg_shape]

        target_loc_shape = (n_batch*sample_per_batch, 2)
        mean_pos_prob_shape = (1,)

        return in_shape, out_shape, [target_loc_shape, mean_pos_prob_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return MultiBoxTarget( \
                self.n_class, self.th_iou, self.th_nms, self.th_neg_nms, 
                self.n_max_label, self.sample_per_label, self.hard_neg_ratio, 
                self.ignore_label, self.variances)

