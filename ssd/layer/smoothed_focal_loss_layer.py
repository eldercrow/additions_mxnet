import mxnet as mx
import numpy as np
from ast import literal_eval


class SmoothedFocalLoss(mx.operator.CustomOp):
    '''
    '''
    def __init__(self, alpha, gamma, th_prob, normalize):
        super(SmoothedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.th_prob = th_prob
        self.normalize = normalize

        self.eps = 1e-14

    def forward(self, is_train, req, in_data, out_data, aux):
        '''
        Just pass the data.
        '''
        self.assign(out_data[0], req[0], in_data[1])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        '''
        Reweight loss according to focal loss.
        '''
        p = mx.nd.pick(in_data[1], in_data[2], axis=1, keepdims=True)

        ce = -mx.nd.log(mx.nd.maximum(p, self.eps))
        sce = -p / self.th_prob - np.log(self.th_prob) + 1

        mask = p > self.th_prob
        sce = mask * ce + (1 - mask) * sce # smoothed cross entropy

        thp = mx.nd.maximum(p, self.th_prob)
        u = 1 - p if self.gamma == 2.0 else mx.nd.power(1 - p, self.gamma - 1.0)
        v = p * (self.gamma * sce + (1 - p) / thp)
        a = (in_data[2] > 0) * self.alpha + (in_data[2] == 0) * (1 - self.alpha)
        gf = v * u * a

        n_class = in_data[1].shape[1]
        alpha = mx.nd.one_hot(mx.nd.reshape(in_data[2], (0, -1)), n_class,
                on_value=1, off_value=0)
        alpha = mx.nd.transpose(alpha, (0, 2, 1))

        g = (in_data[1] - alpha) * gf
        g *= (in_data[2] >= 0)

        if self.normalize:
            g /= mx.nd.sum(in_data[2] > 0).asscalar()

        self.assign(in_grad[0], req[0], g)
        self.assign(in_grad[1], req[1], 0)
        self.assign(in_grad[2], req[2], 0)


@mx.operator.register("smoothed_focal_loss")
class SmoothedFocalLossProp(mx.operator.CustomOpProp):
    '''
    '''
    def __init__(self, alpha=0.25, gamma=2.0, th_prob=0.1, normalize=False):
        #
        super(SmoothedFocalLossProp, self).__init__(need_top_grad=False)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.th_prob = float(th_prob)
        self.normalize = bool(literal_eval(str(normalize)))

    def list_arguments(self):
        return ['cls_pred', 'cls_prob', 'cls_target']

    def list_outputs(self):
        return ['cls_prob']

    def infer_shape(self, in_shape):
        out_shape = [in_shape[0], ]
        return in_shape, out_shape, []

    def create_operator(self, ctx, shapes, dtypes):
        return SmoothedFocalLoss(self.alpha, self.gamma, self.th_prob, self.normalize)