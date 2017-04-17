# pylint: skip-file
import os
# MXNET_CPU_WORKER_NTHREADS must be greater than 1 for custom op to work on CPU
os.environ["MXNET_CPU_WORKER_NTHREADS"] = "4"
import mxnet as mx
import ast

class Ternarize(mx.operator.CustomOp):
    ''' ternarize a given weight '''
    def __init__(self, soft_ternarize):
        #
        super(Ternarize, self).__init__()
        self.soft_ternarize = soft_ternarize
        self.th_ratio = 0.7

    def forward(self, is_train, req, in_data, out_data, aux):
        #
        weight = in_data[0]
        self.th_w = mx.nd.mean(mx.nd.abs(weight), axis=()) * self.th_ratio

        if not soft_ternarize:
            Assign(out_data[0], req[0], 
                    mx.nd.clip(mx.nd.fix(weight / th_w), -1, 1) * th_w)
        else:
            Assign(out_data[0], req[0], mx.nd.clip(weight, -th_w, th_w))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        #
        if not self.soft_ternarize:
            Assign(in_grad[0], req[0], out_grad[0])
        else:
            mask_in = out_data[0] < self.th_w
            mask_in *= out_data[0] > -self.th_w
            Assign(in_grad[0], req[0], out_grad[0] * mask_in)

@mx.operator.register("ternarize")
class TernarizeOp(mx.operator.CustomOpProp):
    def __init__(self, soft_ternarize):
        #
        super(TernarizeOp, self).__init__(need_top_grad=True)
        self.soft_ternarize = bool(ast.literal_eval(soft_ternarize))

    def list_arguments(self):
        return ['weight']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        return in_shape, in_shape, []

    def create_operator(self, ctx, shapes, dtypes):
        return Ternarize(self.soft_ternarize)
