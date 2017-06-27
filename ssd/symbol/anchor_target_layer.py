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
            pred_conv (n nch n_anchor)
            anchor (1 n_anchor 4)
            label(n 5)
        out_data
            target_conv (n*n_sample c)
            target_label (n*n_sample 5)
        aux:
            target_loc (n n_sample)
        '''
        n_batch, nch, n_anchor = in_data[0].shape

        anchors = in_data[2].asnumpy()
        anchors = np.reshape(anchors, (-1, 4))
        anchors = np.transpose(anchors, (1, 0)) # (4 n_anchor)
        labels = in_data[3].asnumpy()

        cls_target = np.full((n_batch, n_anchor), -1, dtype=np.float32)
        bbox_target = np.zeros((n_batch, n_anchor, 4), dtype=np.float32)
        bbox_target_mask = np.zeros((n_batch, n_anchor, 4), dtype=np.float32)

        for i, l in enumerate(labels):
            # compute iou between label and anchors
            ious = _compute_IOU(l, anchors)
            for j in range(self.n_sample):
                midx = np.argmax(ious)
                cls_target[i, midx] = l[0] if j == 0 or ious[midx] >= self.th_iou else self.ignore_label
                bbox_target[i, midx, :] = _compute_target(l, anchors[:, midx], self.variances)
                bbox_target_mask[i, midx, :] = 1
                ious[midx] = 0

        self.assign(out_data[0], req[0], mx.nd.array(cls_target, ctx=in_data[0].context))
        self.assign(out_data[1], req[1], mx.nd.array(bbox_target, ctx=in_data[1].context))
        self.assign(out_data[2], req[2], mx.nd.array(bbox_target_mask, ctx=in_data[1].context))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        for i in range(len(in_grad)):
            self.assign(in_grad[i], req[i], 0)


def _compute_IOU(label, anchors):
    iw = np.maximum(0, \
            np.minimum(label[3], anchors[2, :]) - np.maximum(label[1], anchors[0, :]))
    ih = np.maximum(0, \
            np.minimum(label[4], anchors[3, :]) - np.maximum(label[2], anchors[1, :]))
    I = iw * ih
    U = (label[4] - label[2]) * (label[3] - label[1]) + \
            (anchors[2, :] - anchors[0, :]) * (anchors[3, :] - anchors[1, :])
    
    iou = I / np.maximum((U - I), 1e-08)
    return iou # (num_anchors, )


def _compute_target(label, anchor, variances):
    iw = 1.0 / (anchor[2] - anchor[0])
    ih = 1.0 / (anchor[3] - anchor[1])
    tx = ((label[3] + label[1]) - (anchor[2] + anchor[0])) * 0.5 * iw
    ty = ((label[4] + label[2]) - (anchor[3] + anchor[1])) * 0.5 * ih
    sx = np.log2((label[3] - label[1]) * iw)
    sy = np.log2((label[4] - label[2]) * ih)

    target = np.array((tx, ty, sx, sy)) / variances
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
        return ['cls_preds', 'bbox_preds', 'anchors', 'label']

    def list_outputs(self):
        return ['cls_target', 'bbox_target', 'bbox_target_mask']

    def infer_shape(self, in_shape):
        assert len(in_shape[0]) == 3
        assert len(in_shape[1]) == 3
        out_shape = [(in_shape[0][0], in_shape[0][2]), in_shape[1], in_shape[1]]
        return in_shape, out_shape, []

    def create_operator(self, ctx, shapes, dtypes):
        return AnchorTarget(self.n_sample, self.th_iou, self.ignore_label, self.variances)
