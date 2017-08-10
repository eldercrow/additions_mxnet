import mxnet as mx
import numpy as np
from ast import literal_eval


class ReweightLoss(mx.operator.CustomOp):
    '''
    '''
    def __init__(self, alpha, gamma, normalize):
        super(ReweightLoss, self).__init__()
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
        '''
        n_batch, n_class, n_anchor = in_data[0].shape
        cls_target = mx.nd.reshape(in_data[1], (-1, ))
        t = mx.nd.one_hot(cls_target, n_class, on_value=1, off_value=-1)
        t = mx.nd.reshape(t, (n_batch, n_anchor, n_class))
        t = mx.nd.transpose(t, (0, 2, 1))

        # compute p_t in eq.2 of the paper
        p = in_data[0] * t
        p += (t == -1)

        u = 1 - p - (self.gamma * p * mx.nd.log(mx.nd.maximum(p, self.eps)))
        v = 1 - p if self.gamma == 2.0 else mx.nd.power(1 - p, self.gamma - 1.0)
        g = self.alpha * v * u
        g *= (mx.nd.reshape(in_data[1], (n_batch, 1, -1)) >= 0)
        if self.normalize:
            g /= mx.nd.sum(in_data[1] > 0).asscalar()

        self.assign(in_grad[0], req[0], out_grad[0] * g)
        self.assign(in_grad[1], req[1], 0)


@mx.operator.register("reweight_loss")
class ReweightLossProp(mx.operator.CustomOpProp):
    '''
    '''
    def __init__(self, alpha=0.25, gamma=2.0, normalize=True):
        #
        super(ReweightLossProp, self).__init__(need_top_grad=True)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.normalize = bool(literal_eval(str(normalize)))

    def list_arguments(self):
        return ['cls_prob', 'cls_target']

    def list_outputs(self):
        # return ['cls_target', 'bbox_weight']
        return ['cls_prob']

    def infer_shape(self, in_shape):
        out_shape = [in_shape[0], ]
        return in_shape, out_shape, []

    def create_operator(self, ctx, shapes, dtypes):
        return ReweightLoss(self.alpha, self.gamma, self.normalize)
