import mxnet as mx
import numpy as np

class NMSCandidate(mx.operator.CustomOp):
    '''
    precompute nms candidates.
    '''
    def __init__(self, th_nms, max_cand):
        super(NMSCandidate, self).__init__()
        self.th_nms = th_nms
        self.max_cand = max_cand
        self.nms_cands = None

    def forward(self, is_train, req, in_data, out_data, aux):
        '''
        in_data
            anchors: [n_anchor, 4]
        out_data
            nms_cands: [n_anchor, max_cand]
        '''
        if self.nms_cands is not None:
            self.assign(out_data[0], req[0], self.nms_cands)
            return

        max_neighbor = 0
        anchors = in_data[0].asnumpy()
        area_anchors = (anchors[:, 2] - anchors[:, 0]) * (anchors[:, 3] - anchors[:, 1])
        nms_cands = np.full((anchors.shape[0], self.max_cand), -1, dtype=np.uint32)
        for i, anc in enumerate(anchors):
            iou = _compute_iou(anc, anchors, area_anchors)
            nidx = np.where(iou > self.th_nms)[0]
            np.random.shuffle(nidx)
            eidx = np.minimum(nidx.size, self.max_cand)
            nms_cands[i, :eidx] = nidx[:eidx]
            max_neighbor = np.maximum(max_neighbor, nidx.size)

        self.nms_cands = mx.nd.array(nms_cands, ctx=mx.cpu())
        self.assign(out_data[0], req[0], self.nms_cands)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)

def _compute_iou(anc, anchors, area_anchors):
    iw = np.maximum(0, np.minimum(anc[2], anchors[:, 2]) - np.maximum(anc[0], anchors[:, 0]))
    ih = np.maximum(0, np.minimum(anc[3], anchors[:, 3]) - np.maximum(anc[1], anchors[:, 1]))
    i_ = iw * ih
    u_ = (anc[2] - anc[0]) * (anc[3] - anc[1]) + area_anchors - i_
    return i_ / np.maximum(u_, 1e-08)

@mx.operator.register("nms_candidate")
class NMSCandidateProp(mx.operator.CustomOpProp):
    def __init__(self, th_nms, max_cand):
        super(NMSCandidateProp, self).__init__(need_top_grad=False)
        self.th_nms = float(th_nms)
        self.max_cand = int(max_cand)

    def list_arguments(self):
        return ['anchors']

    def list_outputs(self):
        return ['nms_cands',]

    def infer_shape(self, in_shape):
        n_anchor = in_shape[0][0]
        return in_shape, \
                [[n_anchor, self.max_cand]], \
                []

    def create_operator(self, ctx, shapes, dtypes):
        return MultiBoxPriorPython(self.th_nms, self.max_cand)
