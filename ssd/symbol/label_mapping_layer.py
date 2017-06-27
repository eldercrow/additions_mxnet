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

        label_map = mx.nd.zeros((n_batch, n_anchor), ctx=anchors.context)
        bbox_target = mx.nd.zeros((n_batch, n_anchor, 4), ctx=anchors.context)
        bbox_weight = mx.nd.ones((n_batch, n_anchor), ctx=anchors.context)

        for i, labels in enumerate(labels_all):
            lmap = label_map[i]
            bb_map = bbox_target[i]
            bb_w = bbox_weight[i]
            labels = _get_valid_labels(labels)
            max_iou = mx.nd.zeros((n_anchor, ), ctx=anchors.context)

            for l in labels:
                L = mx.nd.full((n_anchor,), l[0], ctx=anchors.context)
                BB = mx.nd.tile(mx.nd.array(l[1:], ctx=anchors.context), (n_anchor, 1)) # (n_anchor, 4)
                iou = _compute_iou(l[1:], anchors, area_anchors)
                lmask = iou >= max_iou
                lmap = mx.nd.where(lmask, lmap, L)
                bb_map = mx.nd.where(lmask, bb_map, BB)
                max_iou = mx.nd.maximum(iou, max_iou)
            bbox_target[i] = bb_map
            bbox_weight[i] = bb_w * (max_iou > 0.33333)
            label_map[i] = lmap * (max_iou > self.th_iou)

        bbox_target = mx.nd.transpose(bbox_target, axes=(2, 0, 1)) # (4, n_batch, n_anchor)

        self.assign(out_data[0], req[0], label_map)
        self.assign(out_data[1], req[1], \
                _compute_bbox_target(bbox_target, mx.nd.reshape(anchors, shape=(4, 1, -1))))
        self.assign(out_data[2], req[2], mx.nd.tile(bbox_weight, (1, 1, 4)))


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


def _compute_bbox_target(label_map, anchors):
    #
    label_map = mx.nd.transpose(label_map, axes=(2, 0, 1))
    anchors = mx.nd.transpose(anchors, axes=(2, 0, 1))

    aw = anchors[2] - anchors[0]
    ah = anchors[3] - anchors[1]

    bbox_target = mx.nd.zeros_like(label_map)
    bbox_target[0] = ((label_map[2] + label_map[0]) - (anchors[2] - anchors[0])) * 0.5 / aw
    bbox_target[1] = ((label_map[3] + label_map[1]) - (anchors[3] - anchors[1])) * 0.5 / ah
    bbox_target[2] = mx.nd.log2((label_map[2] - label_map[0]) / aw + 1e-08)
    bbox_target[3] = mx.nd.log2((label_map[3] - label_map[1]) / ah + 1e-08)

    return mx.nd.transpose(bbox_target, axes=(1, 2, 0))


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
        return ['label_map', 'bbox_target', 'bbox_weight']


    def infer_shape(self, in_shape):
        assert in_shape[1][2] == 5, 'Label shape should be (n_batch, n_label, 5)'
        n_batch = in_shape[1][0]
        n_anchor = in_shape[0][1]

        label_shape = (n_batch_, n_anchor)
        bbox_shape = (n_batch, n_anchor, 4)
        return in_shape, [label_shape, bbox_shape, bbox_shape], []


    def create_operator(self, ctx, shapes, dtypes):
        return LabelMapping(self, th_iou)
