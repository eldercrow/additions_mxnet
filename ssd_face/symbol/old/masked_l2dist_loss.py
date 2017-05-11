# pylint: skip-file
import os
# MXNET_CPU_WORKER_NTHREADS must be greater than 1 for custom op to work on CPU
os.environ["MXNET_CPU_WORKER_NTHREADS"] = "2"
import mxnet as mx

class MaskedL2Loss(mx.operator.CustomOp):
    #
    def __init__(self, grad_scale=1.0):
        super(MaskedL2Loss, self).__init__()
        self.grad_scale = float(grad_scale)

    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], req[0], in_data[0])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        d = in_data[0] - in_data[1]
        md = d * in_data[2]
        md = md * self.grad_scale
        self.assign(in_grad[0], req[0], md)

@mx.operator.register("maskedl2loss")
class MaskedL2LossProp(mx.operator.CustomOpProp):
    #
    def __init__(self, grad_scale=1.0):
        super(MaskedL2LossProp, self).__init__(need_top_grad=False)
        self.grad_scale = grad_scale

    def list_arguments(self):
        return ['data', 'label', 'mask']
    
    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_shape = in_shape[0]
        mask_shape = (in_shape[0][0], 1)

        output_shape = in_shape[0]

        return [data_shape, label_shape, mask_shape], [output_shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return MaskedL2Loss(self.grad_scale)
