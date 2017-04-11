import os
import mxnet as mx
import numpy as np
# MXNET_CPU_WORKER_NTHREADS must be greater than 1 for custom op to work on CPU
os.environ["MXNET_CPU_WORKER_NTHREADS"] = "4"
from ast import literal_eval as make_tuple

class MultiBoxDetection(mx.operator.CustomOp):
    '''
    python implementation of MultiBoxDetection layer.
    '''
    def __init__(self, n_class, max_detection, th_pos, th_nms, variances):
        #
        super(MultiBoxDetection, self).__init__()
        self.n_class = n_class
        self.th_pos = th_pos
        self.th_nms = th_nms
        self.variances = variances
        self.max_detection = max_detection
        
    def forward(self, is_train, req, in_data, out_data, aux):
        '''
        pick positives, transform bbs, apply nms
        '''
        n_batch = in_data[1].shape[0]
        n_anchor = in_data[1].shape[1]
        n_class = in_data[0].shape[1]

        variances = mx.nd.array(self.variances, ctx=in_data[0].context)
        variances = mx.nd.reshape(variances, (-1, 1))

        probs_cls = mx.nd.reshape(in_data[0], (n_batch, n_anchor, n_class)) # (n_batch, n_anchors, n_classes)
        preds_reg = in_data[1] # (n_batch, n_anchors, 4)
        anchors = in_data[2] # (n_anchors, 4)
        anchors_t = mx.nd.transpose(anchors, axes=(1, 0))
        area_anchors_t = (anchors_t[2] - anchors_t[0]) * (anchors_t[3] - anchors_t[1])

        out_cid = mx.nd.full((n_batch, self.max_detection), -1, ctx=in_data[0].context)
        out_cls = mx.nd.full((n_batch, self.max_detection), -1, ctx=in_data[0].context)
        out_roi = mx.nd.full((n_batch, self.max_detection, 4), -1, ctx=in_data[1].context)
        out_anc = mx.nd.full((n_batch, self.max_detection, 4), -1, ctx=in_data[2].context)

        n_detection = mx.nd.zeros((n_batch,), ctx=in_data[0].context)

        probs_cls = mx.nd.transpose(probs_cls, axes=(2, 0, 1)) # (n_class - 1, n_batch, n_anchor)
        max_probs = mx.nd.max(probs_cls[1:], axis=0) # (n_batch, n_anchor)
        max_cid = mx.nd.argmax(probs_cls[1:], axis=0) + 1 # (n_batch, n_anchor)
        sidx_cls = mx.nd.argsort(max_probs, axis=-1).asnumpy().astype(int)[:, ::-1]
        max_probs = max_probs.asnumpy()

        for nn in range(n_batch):
            pcls = max_probs[nn] # (n_anchor, )
            pcid = max_cid[nn] # (n_anchor, )
            sidx = sidx_cls[nn] # (n_anchor, )
            preg = preds_reg[nn] # (n_anchor, 4)

            ocls = out_cls[nn]
            ocid = out_cid[nn]
            oroi = out_roi[nn]
            oanc = out_anc[nn]
            # gather positive anchors
            pos_idx = np.where(pcls > self.th_pos)[0]
            pos_idx = np.sort(pos_idx)[::-1]
            panc = mx.nd.zeros((len(pos_idx), 4), ctx=in_data[2].context)
            proi = mx.nd.zeros((len(pos_idx), 4), ctx=in_data[2].context)
            for i, p in enumerate(pos_idx):
                proi[i] = preg[p]
                panc[i] = anchors[p]
            proi = _transform_roi(proi, panc, variances)
            # apply nms
            proi_t = mx.nd.transpose(proi, axes=(1, 0))
            area_proi_t = (proi_t[2] - proi_t[0]) * (proi_t[3] - proi_t[1])
            k = 0
            for i in range(len(pos_idx)):
                if pos_idx[i] == -1: 
                    continue
                p = pos_idx[i]
                ocls[k] = pcls[p]
                ocid[k] = pcid[p]
                oroi[k] = proi[i]
                nidx = _nms_anchor(proi[i], proi_t, area_proi_t[i] + area_proi_t, self.th_nms)
                pos_idx[nidx] = -1
                k += 1
                if k == self.max_detection:
                    break
            n_detection[nn] = k

        out_cls = mx.nd.reshape(out_cls, (0, 0, 1))
        out_cid = mx.nd.reshape(out_cid, (0, 0, 1))

        self.assign(out_data[0], req[0], mx.nd.concat(out_cid, out_cls, out_roi, dim=2))
        self.assign(out_data[1], req[1], n_detection)
        
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        pass

def _nms_anchor(anc, anchors_t, U, th_nms):
    iw = mx.nd.minimum(anc[2], anchors_t[2]) - mx.nd.maximum(anc[0], anchors_t[0])
    ih = mx.nd.minimum(anc[3], anchors_t[3]) - mx.nd.maximum(anc[1], anchors_t[1])

    I = mx.nd.maximum(iw, 0) * mx.nd.maximum(ih, 0)
    iou = (I / mx.nd.maximum(U - I, 1e-06)).asnumpy()
    return np.where(iou > th_nms)[0]

def _transform_roi(reg, anc, variances):
    reg_t = mx.nd.transpose(reg, axes=(1, 0))
    reg_t = mx.nd.broadcast_mul(reg_t, variances)
    anc_t = mx.nd.transpose(anc, axes=(1, 0))

    cx = (anc_t[2] + anc_t[0]) * 0.5
    cy = (anc_t[3] + anc_t[1]) * 0.5
    aw = anc_t[2] - anc_t[0]
    ah = anc_t[3] - anc_t[1]
    cx += reg_t[0] * aw
    cy += reg_t[1] * ah
    w = (2.0**reg_t[2]) * aw
    h = (2.0**reg_t[3]) * ah
    reg_t[0] = cx - w / 2.0
    reg_t[1] = cy - h / 2.0
    reg_t[2] = cx + w / 2.0
    reg_t[3] = cy + h / 2.0
    return mx.nd.transpose(reg_t, axes=(1, 0))

@mx.operator.register("multibox_detection")
class MultiBoxDetectionProp(mx.operator.CustomOpProp):
    def __init__(self, n_class, max_detection=1000, th_pos=0.8, th_nms=0.3, variances=(0.1, 0.1, 0.2, 0.2)):
        #
        super(MultiBoxDetectionProp, self).__init__(need_top_grad=True)
        self.n_class = int(n_class)
        self.max_detection = int(max_detection)
        self.th_pos = float(th_pos)
        self.th_nms = float(th_nms)
        if isinstance(variances, str):
            variances = make_tuple(variances)
        self.variances = np.array(variances)

    def list_arguments(self):
        return ['probs_cls', 'preds_reg', 'anchors']

    def list_outputs(self):
        return ['output', 'n_detection']

    def infer_shape(self, in_shape):
        n_batch = in_shape[1][0]
        out_shape = [(n_batch, self.max_detection, 6), (n_batch, )]

        return in_shape, out_shape, []

    def create_operator(self, ctx, shapes, dtypes):
        return MultiBoxDetection(self.n_class, self.max_detection, self.th_pos, self.th_nms, self.variances)
