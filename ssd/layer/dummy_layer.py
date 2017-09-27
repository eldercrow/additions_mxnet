import mxnet as mx
import numpy as np
import cv2

class DummyLayer(mx.operator.CustomOp):
    '''
    '''
    def __init__(self):
        super(DummyLayer, self).__init__()

    def forward(self, is_train, req, in_data, out_data, aux):
        '''
        Just pass the data.
        '''
        img = _to_img(in_data[0][0])

        import ipdb
        ipdb.set_trace()
        self.assign(out_data[0], req[0], in_data[0])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        '''
        Pass the gradient.
        '''
        if self.need_top_grad:
            self.assign(in_grad[0], req[0], out_grad[0])
        else:
            self.assign(in_grad[0], req[0], 0)


def _to_img(data):
    '''
    '''
    img = np.transpose(data.asnumpy(), (1, 2, 0))
    img = img / 255.0 + 0.5
    img = img[:, :, ::-1]
    return img


@mx.operator.register("dummy")
class DummyLayerProp(mx.operator.CustomOpProp):
    '''
    '''
    def __init__(self, need_top_grad=True):
        #
        super(DummyLayerProp, self).__init__(need_top_grad=need_top_grad)

    def list_arguments(self):
        return ['data',]

    def list_outputs(self):
        # return ['cls_target', 'bbox_weight']
        return ['output',]

    def infer_shape(self, in_shape):
        return in_shape, in_shape, []

    def create_operator(self, ctx, shapes, dtypes):
        return DummyLayer()
