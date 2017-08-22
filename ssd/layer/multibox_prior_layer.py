import mxnet as mx
import numpy as np
from collections import Iterable
from ast import literal_eval as make_tuple


class MultiBoxPrior(mx.operator.CustomOp):
    '''
    python alternative of MultiBoxPrior class.
    Will handle anchor box layer in a different way.
    Also I will handle sizes and ratios in a different - like rcnn - way.
    '''
    def __init__(self, sizes, ratios, strides, clip):
        super(MultiBoxPrior, self).__init__()
        self.sizes = sizes
        self.ratios = ratios
        self.strides = strides
        self.clip = clip
        self.anchor_data = None

    def forward(self, is_train, req, in_data, out_data, aux):
        '''
        in_data:
            a conv layer that we will infer the size of outputs.
        out_data:
            anchors (1 num_anchor*4 h w)
        '''
        if self.anchor_data is not None:
            self.assign(out_data[0], req[0], self.anchor_data)
            return

        anchors_all = np.empty((0, 4), dtype=np.float32)

        for ii, (s, r) in enumerate(zip(self.sizes, self.ratios)):
            h = in_data[ii].shape[2]
            w = in_data[ii].shape[3]
            apc = len(s) * len(r)

            if not self.strides:
                stride = (1.0 / w, 1.0 / h)
            else:
                stride = self.strides[ii]
                if not isinstance(stride, Iterable):
                    stride = (stride, stride)

            # compute center positions
            x = (np.arange(w) + 0.5) * stride[0]
            y = (np.arange(h) + 0.5) * stride[1]
            xv, yv = np.meshgrid(x, y)

            # compute heights and widths
            wh = np.zeros((apc, 2))
            k = 0
            for i in s:
                for j in r:
                    j2 = np.sqrt(float(j))
                    wh[k, 0] = (i * j2) * 0.5 # width
                    wh[k, 1] = (i / j2) * 0.5 # height
                    k += 1

            # build anchors
            anchors = np.zeros((h, w, apc, 4), dtype=np.float32)
            for i in range(apc):
                anchors[:, :, i, 0] = xv - wh[i, 0]
                anchors[:, :, i, 1] = yv - wh[i, 1]
                anchors[:, :, i, 2] = xv + wh[i, 0]
                anchors[:, :, i, 3] = yv + wh[i, 1]

            anchors = np.reshape(anchors, (-1, 4))
            anchors_all = np.vstack((anchors_all, anchors))

        if self.clip > 0:
            anchors_all[:, 0::2] = np.minimum(np.maximum(anchors_all[:, 0::2], 0.0), 1.0)
            anchors_all[:, 1::2] = np.minimum(np.maximum(anchors_all[:, 1::2], 0.0), 1.0)
            # anchors_all = np.minimum(np.maximum(anchors_all, 0.0), 1.0)
        anchors_all = np.reshape(anchors_all, (1, -1, 4))
        self.anchor_data = mx.nd.array(anchors_all, ctx=in_data[0].context)
        self.assign(out_data[0], req[0], self.anchor_data)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        for i in range(len(self.sizes)):
            self.assign(in_grad[i], req[i], 0)


@mx.operator.register("multibox_prior")
class MultiBoxPriorProp(mx.operator.CustomOpProp):
    def __init__(self, sizes, ratios, strides=None, clip=False):
        super(MultiBoxPriorProp, self).__init__(need_top_grad=False)
        self.sizes = make_tuple(sizes)
        self.ratios = make_tuple(ratios)
        assert len(self.sizes) == len(self.ratios)
        if strides:
            strides = make_tuple(str(strides))
        self.strides = make_tuple(strides) if strides else None
        import ipdb
        ipdb.set_trace()
        if strides:
            assert len(self.sizes) == len(self.strides)
        self.clip = int(clip)
        # self.strides = [2.0**i for i in range(len(self.sizes))]

    def list_arguments(self):
        return ['ref_conv{}'.format(i) for i in range(len(self.sizes))]

    def list_outputs(self):
        return ['anchors',]

    def infer_shape(self, in_shape):
        n_anchor = 0
        for ii, (s, r) in enumerate(zip(self.sizes, self.ratios)):
            h = in_shape[ii][2]
            w = in_shape[ii][3]
            apc = len(s) * len(r)
            n_anchor += h*w*apc
        return in_shape, [(1, n_anchor, 4),], []

    def create_operator(self, ctx, shapes, dtypes):
        return MultiBoxPrior(self.sizes, self.ratios, self.strides, self.clip)
