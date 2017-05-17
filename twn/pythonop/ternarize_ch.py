# pylint: skip-file
import os
# MXNET_CPU_WORKER_NTHREADS must be greater than 1 for custom op to work on CPU
os.environ["MXNET_CPU_WORKER_NTHREADS"] = "4"
import mxnet as mx
import ast

class TernarizeCh(mx.operator.CustomOp):
    ''' ternarize a given weight '''
    def __init__(self, filterwise):
        #
        super(TernarizeCh, self).__init__()
        self.filterwise = filterwise
        self.th_ratio = 1.0

    def forward(self, is_train, req, in_data, out_data, aux):
        #
        weight = in_data[0]
        abs_weight = mx.nd.abs(weight)
        sum_abs_w = _comp_sum(abs_weight, self.filterwise)
        denom = float(weight.size)
        if self.filterwise == 'filter':
            denom /= weight.shape[0]
        elif self.filterwise == 'channel':
            denom /= weight.shape[1]
        self.th_w = sum_abs_w / denom * self.th_ratio

        amask = abs_weight >= self.th_w
        self.alpha = _comp_sum(abs_weight * amask, self.filterwise) / _comp_sum(amask, self.filterwise)
        self.assign(out_data[0], req[0], mx.nd.sign(weight * amask) * self.alpha)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        #
        self.assign(in_grad[0], req[0], out_grad[0])
        # if not self.filterwise:
        #     self.assign(in_grad[0], req[0], out_grad[0])
        # else:
        #     mask_in = out_data[0] < self.th_w
        #     mask_in *= out_data[0] > -self.th_w
        #     self.assign(in_grad[0], req[0], out_grad[0] * mask_in)

def _comp_sum(w, filterwise):
    # import ipdb
    # ipdb.set_trace()
    if filterwise == 'filter':
        shape_w = w.shape
        if len(shape_w) == 4: # convolution
            sum_w = mx.nd.sum(mx.nd.sum(mx.nd.reshape(w, shape=(0, 0, -1)), axis=2), axis=1)
            sum_w = mx.nd.reshape(sum_w, shape=(-1, 1, 1, 1))
        else:
            sum_w = mx.nd.sum(w, axis=1)
            sum_w = mx.nd.reshape(sum_w, shape=(-1, 1))
    elif filterwise == 'channel':
        shape_w = w.shape
        if len(shape_w) == 4: # convolution
            sum_w = mx.nd.sum(mx.nd.sum(mx.nd.reshape(w, shape=(0, 0, -1)), axis=2), axis=0)
            sum_w = mx.nd.reshape(sum_w, shape=(1, -1, 1, 1))
        else:
            sum_w = mx.nd.sum(w, axis=0)
            sum_w = mx.nd.reshape(sum_w, shape=(1, -1))
    else:
        sum_w = mx.nd.sum(mx.nd.sum(w, axis=1))
    return sum_w

@mx.operator.register("ternarize_ch")
class TernarizeChOp(mx.operator.CustomOpProp):
    def __init__(self, filterwise='all'):
        #
        assert filterwise in ['all', 'filter', 'channel']
        super(TernarizeChOp, self).__init__(need_top_grad=True)
        self.filterwise = str(filterwise)

    def list_arguments(self):
        return ['weight']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        assert len(in_shape[0]) == 4 or len(in_shape[0]) == 2
        return in_shape, in_shape, []

    def infer_type(self, in_type):
        dtype = in_type[0]
        return [dtype], [dtype], []

    def create_operator(self, ctx, shapes, dtypes):
        return TernarizeCh(self.filterwise)

