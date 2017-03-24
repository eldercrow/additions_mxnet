import mxnet as mx
import numpy as np
from ast import literal_eval as make_tuple

class MultiBoxPriorPython(mx.operator.CustomOp):
    ''' 
    python alternative of MultiBoxPrior class.
    Will handle anchor box layer in a different way.
    Also I will handle sizes and ratios in a different - like rcnn - way.
    '''
    def __init__(self, sizes, ratios):
        super(MultiBoxPriorPython, self).__init__()
        self.sizes = sizes
        self.ratios = ratios

    def forward(self, is_train, req, in_data, out_data, aux):
        '''
        in_data:
            a conv layer that we will infer the size of outputs.
        out_data:
            anchors (1 num_anchor*4 h w)
            anchor_scaler (1 num_anchor*4 1 1)
        '''
        h = in_data[0].shape[2]
        w = in_data[0].shape[3]
        n_anchor = len(self.sizes) * len(self.ratios)

        # compute center positions
        x = np.linspace(0.0, 1.0, w+1)
        y = np.linspace(0.0, 1.0, h+1)
        xv, yv = np.meshgrid(x, y)
        xv = xv[0:-1, 0:-1] + 0.5 / w
        yv = yv[0:-1, 0:-1] + 0.5 / h

        # compute heights and widths
        wh = np.zeros((n_anchor, 2))
        k = 0
        for i in self.sizes:
            for j in self.ratios:
                j2 = np.sqrt(float(j))
                wh[k, 0] = i / j2
                wh[k, 1] = i * j2
                k += 1
        # build anchors
        anchors = np.zeros((n_anchor*4, h, w))
        anchor_scales = np.zeros((n_anchor*4))
        for i in range(n_anchor):
            anchors[4*i+0, :, :] = xv
            anchors[4*i+1, :, :] = yv
            anchors[4*i+2, :, :] = wh[i, 0]
            anchors[4*i+3, :, :] = wh[i, 1]
            anchor_scales[(4*i):(4*(i+1))] = np.array((1, 1, 2, 2))
        anchors = np.reshape(anchors, (1, -1, h, w))
        anchor_scales = np.reshape(anchor_scales, (1, -1, 1, 1))

        self.assign(out_data[0], req[0], mx.nd.array(anchors, ctx=in_data[0].context))
        self.assign(out_data[1], req[1], mx.nd.array(anchor_scales, ctx=in_data[0].context))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        pass

@mx.operator.register("multibox_prior_python")
class MultiBoxPriorPythonProp(mx.operator.CustomOpProp):
    def __init__(self, sizes, ratios):
        super(MultiBoxPriorPythonProp, self).__init__(need_top_grad=False)
        self.sizes = make_tuple(sizes)
        self.ratios = make_tuple(ratios)

    def list_arguments(self):
        return ['ref_conv',]

    def list_outputs(self):
        return ['anchors', 'anchor_scales']

    def infer_shape(self, in_shape):
        h = in_shape[0][2]
        w = in_shape[0][3]
        n_anchor = len(self.sizes) * len(self.ratios)
        return [in_shape[0], ], \
                [[1, n_anchor*4, h, w], [1, n_anchor*4, 1, 1]], \
                []

    def create_operator(self, ctx, shapes, dtypes):
        return MultiBoxPriorPython(self.sizes, self.ratios)
