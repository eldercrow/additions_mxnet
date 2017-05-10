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

        probs_cls = mx.nd.reshape(in_data[0], (n_batch, n_anchor, n_class))
        preds_reg = in_data[1]  # (n_batch, n_anchors, 4)
        anchors = in_data[2]  # (n_anchors, 4)
        # im_scale = in_data[3]

        for nn in range(n_batch):
            out_i = out_data[0][nn]
            out_i[:] = 0
            pcls = probs_cls[nn]  # (n_anchors, n_classes)
            preg = preds_reg[nn]  # (n_anchor, 4)
            pcls_t = mx.nd.transpose(pcls, axes=(1, 0))
            max_probs = mx.nd.max(pcls_t[1:], axis=0) #.asnumpy()
            max_cid = mx.nd.argmax(pcls_t[1:], axis=0) #.asnumpy()

            n_detection = int(mx.nd.sum(max_probs > self.th_pos).asscalar())
            if n_detection == 0:
                out_data[1][nn] = 0
                continue

            sidx = mx.nd.argsort(max_probs, is_ascend=False)
            sidx = sidx[:n_detection]
            # sidx = sidx[:self.max_detection]

            oreg_t = mx.nd.transpose(mx.nd.take(preg, sidx))
            oanc_t = mx.nd.transpose(mx.nd.take(anchors, sidx))
            ocls = mx.nd.take(max_probs, sidx)
            ocid = mx.nd.take(max_cid, sidx)
            oreg_t = _transform_roi(oreg_t, oanc_t, self.variances, 0.8)
            oreg = mx.nd.transpose(oreg_t)

            vidx = _nms(oreg_t, self.th_nms)
            n_valid = len(vidx)
            n_detection = np.minimum(n_valid, self.max_detection)
            vidx = vidx[:n_detection]

            for i, vid in enumerate(vidx):
                out_i[i][0] = ocid[vid]
                out_i[i][1] = ocls[vid]
                out_i[i][2:] = oreg[vid]
            # out_i = mx.nd.transpose(out_i)
            # import ipdb
            # ipdb.set_trace()
            # out_i[0][:n_valid] = mx.nd.take(ocid, vidx)
            # out_i[1][:n_valid] = mx.nd.take(ocls, vidx)
            # for i in range(4):
            #     bbi = oreg_t[i]
            #     out_i[i+2][:n_valid] = mx.nd.take(bbi, vidx)
            # out_data[0][nn] = mx.nd.transpose(out_i)

            # out_i = mx.nd.transpose(out_i)
            # out_i[0] = mx.nd.take(max_cid, sidx)
            # out_i[1] = mx.nd.take(max_probs, sidx)
            # out_i[2:] = _transform_roi(oreg_t, oanc_t, self.variances, 0.8) #mx.nd.transpose(mx.nd.take(preg, sidx))
            # # out_i[6:]= mx.nd.transpose(mx.nd.take(anchors, sidx))
            #
            # out_data[0][nn] = mx.nd.transpose(_nms(out_i, n_detection, self.th_nms))


            # out_i = _transform_roi(out_i, 0.8)
            # for i in range(4):
                # out_i[i+2] *= self.variances[i]

            # # gather positive anchors
            # pos_idx = np.where(max_probs > self.th_pos)[0]
            # pos_idx = pos_idx[:np.minimum(len(pos_idx), self.n_detection)]
            # sidx = np.argsort(max_probs[pos_idx])[::-1]
            # pos_idx = pos_idx[sidx]
            # for i, p in enumerate(pos_idx):
            #     if i == out_i.shape[0]:
            #         break
            #     out_i[i][0] = max_cid[p]
            #     out_i[i][1] = max_probs[p]
            #     out_i[i][2:6] = preg[p]
            #     out_i[i][6:] = anchors[p]
            # out_i = mx.nd.transpose(out_i)
            # for i in range(4):
                # out_i[i+2] *= self.variances[i]
            # out_i[6] *= im_scale[nn][1]
            # out_i[7] *= im_scale[nn][0]
            # out_i[8] *= im_scale[nn][1]
            # out_i[9] *= im_scale[nn][0]
            # out_data[0][nn] = mx.nd.transpose(out_i)
            out_data[1][nn] = n_detection
            # out_data[1][nn] = np.minimum(self.max_detection, len(pos_idx))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        pass


def _nms(out_t, th_nms):
    n_detection = out_t.shape[1]
    nms_mask = np.zeros((n_detection,), dtype=np.bool)
    out = mx.nd.transpose(out_t)
    area_out_t = (out_t[2] - out_t[0]) * (out_t[3] - out_t[1])
    for i in range(n_detection):
        if nms_mask[i]: 
            continue
        iw = mx.nd.minimum(out_t[2][i], out_t[2]) - \
                mx.nd.maximum(out_t[0][i], out_t[0])
        ih = mx.nd.minimum(out_t[3][i], out_t[3]) - \
                mx.nd.maximum(out_t[1][i], out_t[1])
        I = mx.nd.maximum(iw, 0) * mx.nd.maximum(ih, 0)
        iou_mask = (I / mx.nd.maximum(area_out_t + area_out_t[i] - I, 1e-06)) > th_nms
        nms_mask = np.logical_or(nms_mask, iou_mask.asnumpy())
        nms_mask[i] = False
    return np.where(nms_mask == False)[0]

# def _nms(out_t, n_detection, th_nms):
#     out = mx.nd.transpose(out_t)
#     area_out_t = (out_t[4] - out_t[2]) * (out_t[5] - out_t[3])
#     for i in range(n_detection):
#         out_t.wait_to_read()
#         if out_t[1][i].asscalar() == 0: 
#             continue
#         iw = mx.nd.minimum(out_t[4][i], out_t[4]) - \
#                 mx.nd.maximum(out_t[2][i], out_t[2])
#         ih = mx.nd.minimum(out_t[5][i], out_t[5]) - \
#                 mx.nd.maximum(out_t[3][i], out_t[3])
#         I = mx.nd.maximum(iw, 0) * mx.nd.maximum(ih, 0)
#         iou_mask = (I / mx.nd.maximum(area_out_t + area_out_t[i] - I, 1e-06)) < th_nms
#         iou_mask[i] = 1
#         out_t[1] *= iou_mask
#     return out_t


def _nms_anchor(anc, anchors_t, U, th_nms):
    iw = mx.nd.minimum(anc[2], anchors_t[2]) - mx.nd.maximum(
        anc[0], anchors_t[0])
    ih = mx.nd.minimum(anc[3], anchors_t[3]) - mx.nd.maximum(
        anc[1], anchors_t[1])

    I = mx.nd.maximum(iw, 0) * mx.nd.maximum(ih, 0)
    iou = (I / mx.nd.maximum(U - I, 1e-06)).asnumpy()
    return np.where(iou > th_nms)[0]


def _transform_roi(reg_t, anc_t, variances, ratio=1.0):
    #
    # reg_t = mx.nd.transpose(reg)
    # anc_t = mx.nd.transpose(anc)
    for i in range(4):
        reg_t[i] *= variances[i]

    cx = (anc_t[0] + anc_t[2]) * 0.5
    cy = (anc_t[1] + anc_t[3]) * 0.5

    aw = anc_t[2] - anc_t[0]
    aw *= ratio
    ah = anc_t[3] - anc_t[1]
    cx += reg_t[0] * aw
    cy += reg_t[1] * ah
    w = (2.0**reg_t[2]) * aw
    h = (2.0**reg_t[3]) * ah
    reg_t[0] = cx - w / 2.0
    reg_t[1] = cy - h / 2.0
    reg_t[2] = cx + w / 2.0
    reg_t[3] = cy + h / 2.0
    return reg_t 

# def _transform_roi(out_t, ratio=1.0):
#     #
#     # out_t = mx.nd.transpose(out_i, axes=(1, 0))
#     for i in range(4):
#         out_t[i + 2] *= self.variances[i]
#
#     cx = (out_t[6] + out_t[8]) * 0.5
#     cy = (out_t[7] + out_t[9]) * 0.5
#
#     aw = out_t[8] - out_t[6]
#     aw *= ratio
#     ah = out_t[9] - out_t[7]
#     cx += out_t[2] * aw
#     cy += out_t[3] * ah
#     w = (2.0**out_t[4]) * aw
#     h = (2.0**out_t[5]) * ah
#     out_t[0] = cx - w / 2.0
#     out_t[1] = cy - h / 2.0
#     out_t[2] = cx + w / 2.0
#     out_t[3] = cy + h / 2.0
#     return out_t # mx.nd.transpose(out_t, axes=(1, 0))


@mx.operator.register("multibox_detection")
class MultiBoxDetectionProp(mx.operator.CustomOpProp):
    def __init__(self,
                 n_class,
                 max_detection=1000,
                 th_pos=0.5,
                 th_nms=0.3333,
                 variances=(0.1, 0.1, 0.2, 0.2)):
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
        aux_shape = []
        return in_shape, out_shape, aux_shape

    def create_operator(self, ctx, shapes, dtypes):
        return MultiBoxDetection(self.n_class, self.max_detection, self.th_pos,
                                 self.th_nms, self.variances)
