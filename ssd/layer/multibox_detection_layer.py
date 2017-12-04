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

    def __init__(self, max_detection, th_pos, th_nms, per_cls_reg, variances):
        #
        super(MultiBoxDetection, self).__init__()
        self.th_pos = th_pos
        self.th_nms = th_nms
        self.per_cls_reg = per_cls_reg
        self.variances = variances
        self.max_detection = max_detection

    def forward(self, is_train, req, in_data, out_data, aux):
        '''
        pick positives, transform bbs, apply nms
        '''
        n_batch, n_class, n_anchor = in_data[0].shape

        probs_cls = in_data[0]
        preds_reg = mx.nd.reshape(in_data[1], (n_batch, -1, 4))  # (n_batch, n_anchors, 4)
        anchors = mx.nd.reshape(in_data[2], (-1, 4))  # (n_anchors, 4)
        # im_scale = in_data[3]

        for nn in range(n_batch):
            out_i = mx.nd.transpose(out_data[0][nn], (1, 0))
            # out_i[:] = 0
            pcls = probs_cls[nn]  # (n_classes, n_anchors)
            preg = preds_reg[nn]  # (n_anchor, 4)

            if n_class == 1:
                iidx = mx.nd.reshape(pcls > self.th_pos, (-1,))
                out_i[0] = iidx - 1
                out_i[1][:] = mx.nd.reshape(pcls, (-1,))
            else:
                out_i[1] = mx.nd.max(pcls, axis=0)
                iidx = out_i[1] > self.th_pos
                out_i[0] = iidx * (mx.nd.argmax(pcls, axis=0) + 1) - 1
                if self.per_cls_reg:
                    import ipdb
                    ipdb.set_trace()
                    preg = mx.nd.reshape(preg, (n_anchor, -1, 4)) # (n_anchor, n_class, 4)
                    cids = mx.nd.argmax(pcls, axis=0) # (n_anchor)
                    cids = mx.nd.tile(mx.nd.reshape(cids, (n_anchor, 1, 1)), (1, 1, 4)) # (n_anchor, 1, 4)
                    preg = mx.nd.pick(preg, cids) # (n_anchor, 4)
            iidx = mx.nd.array(np.where(iidx.asnumpy())[0])
            out_i[2:] = mx.nd.transpose(_transform_roi(preg, anchors, iidx, self.variances, 1.0))
            # out_i[2:] = _transform_roi( \
            #         mx.nd.transpose(preg), mx.nd.transpose(anchors), iidx, self.variances, 1.0)
            out_data[0][nn] = mx.nd.transpose(out_i, (1, 0))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        pass


def _nms(out_t, th_nms):
    ''' GPU nms '''
    n_detection = out_t.shape[1]
    nms_mask = np.zeros((n_detection,), dtype=np.bool)
    out = mx.nd.transpose(out_t)
    area_out_t = (out_t[2] - out_t[0]) * (out_t[3] - out_t[1])
    for i in range(n_detection-1):
        if nms_mask[i]:
            continue
        iw = mx.nd.minimum(out_t[2][i], out_t[2]) - \
                mx.nd.maximum(out_t[0][i], out_t[0])
        ih = mx.nd.minimum(out_t[3][i], out_t[3]) - \
                mx.nd.maximum(out_t[1][i], out_t[1])
        I = mx.nd.maximum(iw, 0) * mx.nd.maximum(ih, 0)
        iou_mask = (I / mx.nd.maximum(area_out_t + area_out_t[i] - I, 1e-06)) > th_nms
        nidx = np.where(iou_mask.asnumpy())[0]
        nms_mask[nidx] = True
        nms_mask[i] = False
    return np.where(nms_mask == False)[0]


def _transform_roi(reg, anc, iidx, variances, ratio=1.0):
    #
    if iidx.size == 0:
        return reg
    reg_t = mx.nd.transpose(mx.nd.take(reg, iidx))
    anc_t = mx.nd.transpose(mx.nd.take(anc, iidx))

    for i in range(4):
        reg_t[i] *= variances[i]

    cx = (anc_t[0] + anc_t[2]) * 0.5
    cy = (anc_t[1] + anc_t[3]) * 0.5

    aw = anc_t[2] - anc_t[0]
    ah = anc_t[3] - anc_t[1]
    aw *= ratio
    cx += reg_t[0] * aw
    cy += reg_t[1] * ah
    w = mx.nd.exp(reg_t[2]) * aw * 0.5
    h = mx.nd.exp(reg_t[3]) * ah * 0.5
    reg_t[0] = cx - w
    reg_t[1] = cy - h
    reg_t[2] = cx + w
    reg_t[3] = cy + h

    reg_tt = mx.nd.transpose(reg_t)
    for i, j in enumerate(iidx.asnumpy().astype(int)):
        reg[j] = reg_tt[i]
    return reg
# def _transform_roi(reg_t, anc_t, variances, ratio=1.0):
#     #
#     # reg_t = mx.nd.transpose(reg)
#     # anc_t = mx.nd.transpose(anc)
#     for i in range(4):
#         reg_t[i] *= variances[i]
#
#     cx = (anc_t[0] + anc_t[2]) * 0.5
#     cy = (anc_t[1] + anc_t[3]) * 0.5
#
#     aw = anc_t[2] - anc_t[0]
#     ah = anc_t[3] - anc_t[1]
#     aw *= ratio
#     cx += reg_t[0] * aw
#     cy += reg_t[1] * ah
#     w = (2.0**reg_t[2]) * aw * 0.5
#     h = (2.0**reg_t[3]) * ah * 0.5
#     reg_t[0] = cx - w
#     reg_t[1] = cy - h
#     reg_t[2] = cx + w
#     reg_t[3] = cy + h
#     return reg_t


@mx.operator.register("multibox_detection")
class MultiBoxDetectionProp(mx.operator.CustomOpProp):
    def __init__(self,
                 max_detection=1000,
                 th_pos=0.5,
                 th_nms=0.35,
                 per_cls_reg=False,
                 variances=(0.1, 0.1, 0.2, 0.2)):
        #
        super(MultiBoxDetectionProp, self).__init__(need_top_grad=False)
        self.max_detection = int(max_detection)
        self.th_pos = float(th_pos)
        self.th_nms = float(th_nms)
        self.per_cls_reg = bool(make_tuple(str(per_cls_reg)))
        if isinstance(variances, str):
            variances = make_tuple(variances)
        self.variances = np.array(variances)

    def list_arguments(self):
        return ['probs_cls', 'preds_reg', 'anchors']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        n_batch, _, n_anchor = in_shape[0]
        out_shape = [(n_batch, n_anchor, 6)]
        return in_shape, out_shape, []

    def create_operator(self, ctx, shapes, dtypes):
        return MultiBoxDetection(self.max_detection, self.th_pos,
                                 self.th_nms, self.per_cls_reg, self.variances)
