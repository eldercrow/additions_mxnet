import mxnet as mx
import numpy as np


class ReweightLoss(mx.operator.CustomOp):
    '''
    '''
    def __init__(self, n_class, alpha, gamma, normalize):
        super(ReweightLoss, self).__init__()
        self.n_class = n_class
        self.alpha = alpha
        self.gamma = gamma
        self.normalize = normalize

        self.eps = 1e-08

    def forward(self, is_train, req, in_data, out_data, aux):
        '''
        Just pass the data.
        '''
        self.assign(out_data[0], req[0], in_data[0])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        '''
        Reweight loss according to focal loss.
        For now we use gamma == 2 case, JUST FOR TEST.
        '''
        p = mx.nd.pick(in_data[0], in_data[1], axis=1, keepdims=True)
        u = 1 - p - (p * mx.nd.log(mx.nd.maximum(p, self.eps)))
        g = self.alpha * (1 - p) * u
        if self.normalize:
            g /= mx.nd.sum(in_data[1] > 0).asscalar()
        g *= (in_data[1] >= 0)

        self.assign(in_grad[0], req[0], out_grad[0] * g)
        self.assign(in_grad[1], req[1], 0)


@mx.operator.register("reweight_loss")
class ReweightLossProp(mx.operator.CustomOpProp):
    '''
    '''
    def __init__(self, n_class, alpha=0.25, gamma=2.0, normalize=True):
        #
        super(ReweightLossProp, self).__init__(need_top_grad=True)
        self.n_class = n_class
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.normalize = normalize

        assert gamma == 2.0

    def list_arguments(self):
        return ['cls_prob', 'cls_target']

    def list_outputs(self):
        # return ['cls_target', 'bbox_weight']
        return ['cls_prob']

    def infer_shape(self, in_shape):
        out_shape = [in_shape[0], ]
        return in_shape, out_shape, []

    def create_operator(self, ctx, shapes, dtypes):
        return ReweightLoss(self.n_class, self.alpha, self.gamma, self.normalize)
