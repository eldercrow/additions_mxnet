# pylint: skip-file
import os
# MXNET_CPU_WORKER_NTHREADS must be greater than 1 for custom op to work on CPU
os.environ["MXNET_CPU_WORKER_NTHREADS"] = "4"
import mxnet as mx
import ast

class TernarizeCh(mx.operator.CustomOp):
    ''' ternarize a given weight '''
    def __init__(self):
        #
        super(TernarizeCh, self).__init__()
        self.soft_ternarize = soft_ternarize
        self.th_ratio = 1.0 # arxiv: 1705.01462

    def forward(self, is_train, req, in_data, out_data, aux):
        #
        weight = in_data[0]
        abs_weight = mx.nd.abs(weight)
        self.th_w = _comp_sum(abs_weight) / float(weight.size / weight.shape[0]) * self.th_ratio

        amask = abs_weight >= self.th_w
        self.alpha = _comp_sum(abs_weight * amask) / _comp_sum(amask)
        self.assign(out_data[0], req[0], mx.nd.sign(weight * amask) * self.alpha)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        #
        self.assign(in_grad[0], req[0], out_grad[0])
        # if not self.soft_ternarize:
        #     self.assign(in_grad[0], req[0], out_grad[0])
        # else:
        #     mask_in = out_data[0] < self.th_w
        #     mask_in *= out_data[0] > -self.th_w
        #     self.assign(in_grad[0], req[0], out_grad[0] * mask_in)

def _comp_sum(w):
    # per filter sum
    if w.ndim == 4:
        return mx.nd.reshape(mx.nd.sum(w, axis=(1, 2, 3)), shape=(-1, 1, 1, 1))
    else:
        return mx.nd.reshape(mx.nd.sum(w, axis=(1,)), shape=(-1, 1))

@mx.operator.register("ternarize_ch")
class TernarizeChOp(mx.operator.CustomOpProp):
    def __init__(self):
        #
        super(TernarizeChOp, self).__init__(need_top_grad=True)

    def list_arguments(self):
        return ['weight']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        return in_shape, in_shape, []

    def infer_type(self, in_type):
        dtype = in_type[0]
        return [dtype], [dtype], []

    def create_operator(self, ctx, shapes, dtypes):
        return TernarizeCh()
