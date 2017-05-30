# pylint: skip-file
import os
# MXNET_CPU_WORKER_NTHREADS must be greater than 1 for custom op to work on CPU
os.environ["MXNET_CPU_WORKER_NTHREADS"] = "4"
import mxnet as mx
import ast

class TernarizeCh(mx.operator.CustomOp):
    ''' ternarize a given weight '''
    def __init__(self, filterwise, th_ratio):
        #
        super(TernarizeCh, self).__init__()
        self.filterwise = filterwise
        self.th_ratio = th_ratio

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
        if w.ndim == 4: # convolution
            sum_w = mx.nd.sum(w, axis=(2,3), keepdims=True)
            sum_w = mx.nd.sum(w, axis=(1,), keepdims=True)
        else:
            sum_w = mx.nd.sum(w, axis=(1,), keepdims=True)
    elif filterwise == 'channel':
        if w.ndim == 4: # convolution
            sum_w = mx.nd.sum(w, axis=(2,3), keepdims=True)
            sum_w = mx.nd.sum(w, axis=(0,), keepdims=True)
        else:
            sum_w = mx.nd.sum(w, axis=(0,), keepdims=True)
    else:
        sum_w = mx.nd.sum(mx.nd.sum(w, axis=1))
    return sum_w

@mx.operator.register("ternarize_ch")
class TernarizeChOp(mx.operator.CustomOpProp):
    def __init__(self, filterwise='all', th_ratio=1.0):
        #
        assert filterwise in ['all', 'filter', 'channel']
        super(TernarizeChOp, self).__init__(need_top_grad=True)
        self.filterwise = str(filterwise)
        self.th_ratio = float(th_ratio)

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
        return TernarizeCh(self.filterwise, self.th_ratio)

