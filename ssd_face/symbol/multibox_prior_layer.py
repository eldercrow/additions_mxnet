import mxnet as mx
import numpy as np
from ast import literal_eval as make_tuple

class MultiBoxPriorPython(mx.operator.CustomOp):
    ''' 
    python alternative of MultiBoxPrior class.
    Will handle anchor box layer in a different way.
    Also I will handle sizes and ratios in a different - like rcnn - way.
    '''
    def __init__(self, sizes, ratios, clip):
        super(MultiBoxPriorPython, self).__init__()
        self.sizes = sizes
        self.ratios = ratios
        self.clip = clip
        self.anchor_data = None

    def forward(self, is_train, req, in_data, out_data, aux):
        '''
        in_data:
            a conv layer that we will infer the size of outputs.
        out_data:
            anchors (1 num_anchor*4 h w)
            anchor_scaler (1 num_anchor*4 1 1)
        '''
        if self.anchor_data is not None:
            self.assign(out_data[0], req[0], self.anchor_data)
            return

        anchors_all = np.empty((0, 4), dtype=np.float32)

        for ii, (s, r) in enumerate(zip(self.sizes, self.ratios)):
            h = in_data[ii].shape[2]
            w = in_data[ii].shape[3]
            apc = len(s) * len(r)

            # compute center positions
            x = np.linspace(0.0, 1.0, w+1)
            y = np.linspace(0.0, 1.0, h+1)
            xv, yv = np.meshgrid(x, y)
            xv = xv[:-1, :-1] + 0.5 / w
            yv = yv[:-1, :-1] + 0.5 / h

            # compute heights and widths
            wh = np.zeros((apc, 2))
            k = 0
            for i in s:
                for j in r:
                    j2 = np.sqrt(float(j))
                    wh[k, 0] = i * j2 / 2.0
                    wh[k, 1] = i / j2 / 2.0
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
            anchors_all = np.minimum(np.maximum(anchors_all, 0.0), 1.0)
        self.anchor_data = mx.nd.array(anchors_all, ctx=in_data[0].context)
        self.assign(out_data[0], req[0], self.anchor_data)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        for i in range(len(self.sizes)):
            self.assign(in_grad[i], req[i], 0)

@mx.operator.register("multibox_prior_python")
class MultiBoxPriorPythonProp(mx.operator.CustomOpProp):
    def __init__(self, sizes, ratios, clip):
        super(MultiBoxPriorPythonProp, self).__init__(need_top_grad=False)
        self.sizes = make_tuple(sizes)
        self.ratios = make_tuple(ratios)
        assert len(self.sizes) == len(self.ratios)
        self.clip = int(clip)

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
        return in_shape, \
                [[n_anchor, 4]], \
                []

    def create_operator(self, ctx, shapes, dtypes):
        return MultiBoxPriorPython(self.sizes, self.ratios, self.clip)