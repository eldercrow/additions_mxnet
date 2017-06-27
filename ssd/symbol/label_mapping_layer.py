import mxnet as mx
import numpy as np


class LabelMapping(mx.operator.CustomOp):
    '''
    Convert labels (n_batch, n_max_label, 5) into (n_batch, n_anchor).
    '''
    def __init__(self, th_iou):
        super(LabelMapping, self).__init__()
        self.th_iou = th_iou


    def forward(self, is_train, req, in_data, out_data, aux):
        '''
        '''
        anchors = mx.nd.reshape(in_data[0], shape=(-1, 4))
        anchors = mx.nd.transpose(anchors) # (4, n_anchor)
        area_anchors = (anchors[2] - anchors[0]) * (anchors[3] - anchors[1])
        labels_all = in_data[1].asnumpy()

        n_batch = labels_all.shape[0]
        n_anchor = anchors.shape[1]

        label_map = mx.nd.zeros((n_batch, n_anchor, 5), ctx=anchors.context)

        for i, labels in enumerate(labels_all):
            lmap = label_map[i]
            labels = _get_valid_labels(labels)
            max_iou = mx.nd.full((n_anchor, ), self.th_iou, ctx=anchors.context)

            for l in labels:
                L = mx.nd.tile(mx.nd.array((l), ctx=anchors.context), (n_anchor, 1))
                iou = _compute_iou(l[1:], anchors, area_anchors)
                lmask = iou >= max_iou
                lmap = mx.nd.where(lmask, lmap, L)
                lmap[lmask] = l[0]
                max_iou = mx.nd.maximum(iou, max_iou)

        self.assign(out_data[0], req[0], label_map)


    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        for i in range(len(in_grad)):
            self.assign(in_grad[i], req[i], 0)


def _get_valid_labels(labels):
    #
    n_valid_label = 0
    for label in labels:
        if np.all(label == -1.0):
            break
        n_valid_label += 1
    return labels[:n_valid_label, :]


def _compute_iou(label, anchors_t, area_anchors_t):
    #
    iw = mx.nd.minimum(label[2], anchors_t[2]) - mx.nd.maximum(label[0], anchors_t[0])
    ih = mx.nd.minimum(label[3], anchors_t[3]) - mx.nd.maximum(label[1], anchors_t[1])
    I = mx.nd.maximum(iw, 0) * mx.nd.maximum(ih, 0)
    U = (label[3] - label[1]) * (label[2] - label[0]) + area_anchors_t

    iou = I / mx.nd.maximum((U - I), 1e-08)
    return iou # (num_anchors, )


@mx.operator.register("label_mapping")
class LabelMappingProp(mx.operator.CustomOpProp):
    '''
    '''
    def __init__(self, th_iou=0.5):
        '''
        '''
        super(LabelMappingProp, self).__init__(need_top_grad=False)
        self.th_iou = float(th_iou)


    def list_arguments(self):
        return ['anchors', 'label']


    def list_outputs(self):
        return ['label_map']


    def infer_shape(self, in_shape):
        assert in_shape[1][2] == 5, 'Label shape should be (n_batch, n_label, 5)'
        n_batch = in_shape[1][0]
        n_anchor = in_shape[0][1]

        return in_shape, [(n_batch, n_anchor, 5)], []


    def create_operator(self, ctx, shapes, dtypes):
        return LabelMapping(self, th_iou)
