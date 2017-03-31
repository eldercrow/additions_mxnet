import mxnet as mx
import numpy as np

class AnchorTarget(mx.operator.CustomOp):
    ''' 
    This class is not related to anchor target layer in rcnn.
    Get a prediction data (nchw), anchor parameters and label (class and bb), 
    and output a small subset of prediction data (nd) and its labels (class and regression target).
    '''
    def __init__(self, n_sample, th_iou, ignore_label=-1, variances=(0.1, 0.1, 0.2, 0.2)):
        super(AnchorTarget, self).__init__()
        self.n_sample = n_sample
        self.th_iou = th_iou
        self.ignore_label = ignore_label
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

        target_conv = mx.nd.zeros((n_batch * self.n_sample, nch), ctx=in_data[0].context)
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
                if iou[j] < self.th_iou:
                    target_labels[k, :] = -1
                else:
                    target_labels[k, :] = _compute_target(l, anchors[:, sidx[j]], self.variances)
                k += 1
            target_locs[i, :] = sidx

        # save results
        self.assign(aux[0], 'write', mx.nd.array(target_locs))

        self.assign(out_data[0], req[0], target_conv)
        self.assign(out_data[1], req[1], mx.nd.array(target_labels[:, 0], ctx=in_data[0].context))
        self.assign(out_data[2], req[2], mx.nd.array(target_labels[:, 1:], ctx=in_data[0].context))
        self.assign(out_data[3], req[3], mx.nd.array(target_labels[:, 0:1] > 0, ctx=in_data[0].context))

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
    tx = ((label[3] + label[1]) - (anchor[2] + anchor[0])) * 0.5 
    ty = ((label[4] + label[2]) - (anchor[3] + anchor[1])) * 0.5 
    sx = np.log2((label[3] - label[1]) * iw)
    sy = np.log2((label[4] - label[2]) * ih)

    target = np.array((label[0], tx, ty, sx, sy))
    target[1:] /= variances
    return target

@mx.operator.register("anchor_target")
class AnchorTargetProp(mx.operator.CustomOpProp):
    def __init__(self, n_sample=3, th_iou=0.5, ignore_label=-1, variances=(0.1, 0.1, 0.2, 0.2)):
        super(AnchorTargetProp, self).__init__(need_top_grad=True)
        self.n_sample = int(n_sample)
        self.th_iou = float(th_iou)
        self.ignore_label = int(ignore_label)
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
        pred_target_shape = (n_batch*self.n_sample, nch)
        target_cls_shape = (n_batch*self.n_sample, )
        target_reg_shape = (n_batch*self.n_sample, 4)
        mask_reg_shape = (n_batch*self.n_sample, 1)
        target_loc_shape = (n_batch, self.n_sample)

        return [in_shape[0], in_shape[1], in_shape[2]], \
                [pred_target_shape, target_cls_shape, target_reg_shape, mask_reg_shape], \
                [target_loc_shape, ]

    def create_operator(self, ctx, shapes, dtypes):
        return AnchorTarget(self.n_sample, self.th_iou, self.ignore_label, self.variances)
