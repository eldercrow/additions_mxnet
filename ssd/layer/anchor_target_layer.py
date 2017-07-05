import mxnet as mx
import numpy as np
from ast import literal_eval as make_tuple

class AnchorTarget(mx.operator.CustomOp):
    '''
    This class is not related to anchor target layer in rcnn,
    Get a prediction data (nchw), anchor parameters and label (class and bb),
    and output a small subset of prediction data (nd) and its labels (class and regression target).
    '''
    def __init__(self, n_class, n_sample, th_iou, ignore_label, per_cls_reg, variances):
        super(AnchorTarget, self).__init__()
        self.n_class = n_class
        self.n_sample = n_sample
        self.th_iou = th_iou
        self.ignore_label = ignore_label
        self.per_cls_reg = per_cls_reg
        self.variances = variances

    def forward(self, is_train, req, in_data, out_data, aux):
        '''
        in_data:
            pred_conv (n n_anchor nch)
            anchor (n_anchor 4)
            label(n 5)
        out_data
            target_conv (n*n_sample c)
            target_label (n*n_sample 5)
        aux:
            target_loc (n n_sample)
        '''
        n_batch, n_anchor, nch = in_data[0].shape

        pred_conv = in_data[0]
        anchors = in_data[1].asnumpy()
        # anchors = np.reshape(anchors, (-1, 4))
        anchors = np.transpose(anchors, (1, 0)) # (4 n_anchor)
        labels = in_data[2].asnumpy()

        nch_reg = 4 * self.n_class if self.per_cls_reg else 4

        target_conv = mx.nd.zeros((n_batch * self.n_sample, nch), ctx=in_data[0].context)
        target_cls = np.zeros((n_batch * self.n_sample, ))
        target_reg = np.zeros((n_batch * self.n_sample, nch_reg))
        mask_reg = np.zeros_like(target_reg)
        target_labels = np.zeros((n_batch * self.n_sample, 5))
        target_locs = np.zeros((n_batch, self.n_sample))

        # default_loc = np.array((hh*ww/2 + ww/2, ))
        k = 0
        for i, l in enumerate(labels):
            # compute iou between label and anchors
            ious = _compute_IOU(l, anchors)
            sidx = np.argsort(ious)[::-1]
            sidx = sidx[:self.n_sample]
            iou = ious[sidx]
            # copy target data
            preds = pred_conv[i] # (n_anchor c)
            for j in range(self.n_sample):
                # conv data
                target_conv[k] = preds[sidx[j]]
                # label
                if iou[j] > self.th_iou:
                    target_cls[k] = l[0]
                else:
                    target_cls[k] = -1
                if iou[j] > 1.0 / 3.0 and l[0] > 0:
                    r, m = _compute_target(l[1:], anchors[:, sidx[j]], self.variances)
                    if self.per_cls_reg:
                        r, m = _expand_target(r, int(l[0]), self.n_class)
                    target_reg[k, :] = r
                    mask_reg[k, :] = m
                k += 1
            target_locs[i, :] = sidx

        # save results
        self.assign(aux[0], 'write', mx.nd.array(target_locs))

        self.assign(out_data[0], req[0], target_conv)
        self.assign(out_data[1], req[1], mx.nd.array(target_cls, ctx=in_data[0].context))
        self.assign(out_data[2], req[2], mx.nd.array(target_reg, ctx=in_data[0].context))
        self.assign(out_data[3], req[3], mx.nd.array(mask_reg, ctx=in_data[0].context))

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
                gradf[l] = out_grad[0][k]
                k += 1
        self.assign(in_grad[0], req[0], grad_all)
        self.assign(in_grad[1], req[1], 0)
        self.assign(in_grad[2], req[2], 0)

def _compute_IOU(label, anchors):
    iw = np.maximum(0, \
            np.minimum(label[3], anchors[2, :]) - np.maximum(label[1], anchors[0, :]))
    ih = np.maximum(0, \
            np.minimum(label[4], anchors[3, :]) - np.maximum(label[2], anchors[1, :]))
    I = iw * ih
    U = (label[4] - label[2]) * (label[3] - label[1]) + \
            (anchors[2, :] - anchors[0, :]) * (anchors[3, :] - anchors[1, :])

    iou = I / np.maximum((U - I), 0.000001)

    return iou # (num_anchors, )

def _compute_target(label, anchor, variances):
    iw = 1.0 / (anchor[2] - anchor[0])
    ih = 1.0 / (anchor[3] - anchor[1])
    tx = ((label[2] + label[0]) - (anchor[2] + anchor[0])) * 0.5 * iw
    ty = ((label[3] + label[1]) - (anchor[3] + anchor[1])) * 0.5 * ih
    sx = np.log2((label[2] - label[0]) * iw)
    sy = np.log2((label[3] - label[1]) * ih)

    target = np.array((tx, ty, sx, sy)) / variances
    mask = np.ones((4,))
    return target, mask

def _expand_target(target, cid, n_class):
    r = np.zeros((n_class * 4,))
    m = np.zeros_like(r)
    sidx = cid * 4
    r[sidx:sidx+4] = target
    m[sidx:sidx+4] = 1
    return r, m

@mx.operator.register("anchor_target")
class AnchorTargetProp(mx.operator.CustomOpProp):
    def __init__(self, n_class, n_sample=3, th_iou=0.5, ignore_label=-1,
            per_cls_reg=False, variances=(0.1, 0.1, 0.2, 0.2)):
        #
        super(AnchorTargetProp, self).__init__(need_top_grad=True)
        self.n_class = int(n_class)
        self.n_sample = int(n_sample)
        self.th_iou = float(th_iou)
        self.ignore_label = int(ignore_label)
        self.per_cls_reg = bool(make_tuple(str(per_cls_reg)))
        if isinstance(variances, str):
            variances = make_tuple(variances)
        self.variances = np.array(variances).astype(float)

    def list_arguments(self):
        return ['pred_conv', 'anchors', 'label']

    def list_outputs(self):
        return ['pred_target', 'target_cls', 'target_reg', 'mask_reg']

    def list_auxiliary_states(self):
        return ['target_loc_weight', ]

    def infer_shape(self, in_shape):
        n_batch = in_shape[0][0]
        nch = in_shape[0][2]
        nch_reg = self.n_class * 4 if self.per_cls_reg else 4
        pred_target_shape = (n_batch*self.n_sample, nch)
        target_cls_shape = (n_batch*self.n_sample, )
        target_reg_shape = (n_batch*self.n_sample, nch_reg)
        mask_reg_shape = (n_batch*self.n_sample, nch_reg)
        target_loc_shape = (n_batch, self.n_sample)

        return [in_shape[0], in_shape[1], in_shape[2]], \
                [pred_target_shape, target_cls_shape, target_reg_shape, mask_reg_shape], \
                [target_loc_shape, ]

    def create_operator(self, ctx, shapes, dtypes):
        return AnchorTarget(self.n_class, self.n_sample, self.th_iou, self.ignore_label, 
                self.per_cls_reg, self.variances)
