import mxnet as mx
import numpy as np


class DummyLayer(mx.operator.CustomOp):
    '''
    '''
    def __init__(self):
        super(DummyLayer, self).__init__()

    def forward(self, is_train, req, in_data, out_data, aux):
        '''
        Just pass the data.
        '''
        self.assign(out_data[0], req[0], in_data[0])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        '''
        Pass the gradient.
        '''
        import ipdb
        ipdb.set_trace()
        self.assign(in_grad[0], req[0], out_grad[0])


@mx.operator.register("dummy")
class DummyLayerProp(mx.operator.CustomOpProp):
    '''
    '''
    def __init__(self):
        #
        super(DummyLayerProp, self).__init__(need_top_grad=True)

    def list_arguments(self):
        return ['data',]

    def list_outputs(self):
        # return ['cls_target', 'bbox_weight']
        return ['output',]

    def infer_shape(self, in_shape):
        return in_shape, in_shape, []

    def create_operator(self, ctx, shapes, dtypes):
        return DummyLayer()
