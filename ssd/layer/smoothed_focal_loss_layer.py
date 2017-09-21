import mxnet as mx
import numpy as np
import logging
from ast import literal_eval


class SmoothedFocalLoss(mx.operator.CustomOp):
    '''
    '''
    def __init__(self, alpha, gamma, th_prob, w_reg, normalize):
        super(SmoothedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.th_prob = th_prob
        self.w_reg = w_reg
        self.normalize = normalize

        self.eps = np.finfo(np.float32).eps
        self.inited = False

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
        # p = mx.nd.maximum(p, self.eps)

        # th_prob = self.th_prob
        # if not self.inited:
        #     self.inited = True
        #     import ipdb
        #     ipdb.set_trace()
        th_prob = in_data[3].asscalar()

        ce = -mx.nd.log(mx.nd.maximum(p, self.eps))
        sce = -p / th_prob - np.log(th_prob) + 1

        mask = p > th_prob
        sce = mask * ce + (1 - mask) * sce # smoothed cross entropy

        thp = mx.nd.maximum(p, th_prob)
        u = 1 - p if self.gamma == 2.0 else mx.nd.power(1 - p, self.gamma - 1.0)
        v = p * self.gamma * sce + (p / thp) * (1 - p)
        a = (in_data[2] > 0) * self.alpha + (in_data[2] == 0) * (1 - self.alpha)
        gf = v * u * a

        n_class = in_data[1].shape[1]
        alpha = mx.nd.one_hot(mx.nd.reshape(in_data[2], (0, -1)), n_class,
                on_value=1, off_value=0)
        alpha = mx.nd.transpose(alpha, (0, 2, 1))

        g = (in_data[1] - alpha) * gf
        g *= (in_data[2] >= 0)

        g_th = mx.nd.minimum(p, th_prob) / th_prob / th_prob - 1.0 / th_prob
        g_th *= mx.nd.power(1 - p, self.gamma)
        g_th = mx.nd.sum(g_th) + in_data[2].size * th_prob * 2.0 * self.w_reg

        if self.normalize:
            g /= mx.nd.sum(in_data[2] > 0).asscalar()
            g_th /= mx.nd.sum(in_data[2] > 0).asscalar() #in_data[2].size
        if mx.nd.uniform(0, 1, (1,)).asscalar() < 0.001:
            logging.getLogger().info('Current th_prob for smoothed CE: {}'.format(th_prob))

        self.assign(in_grad[0], req[0], g)
        self.assign(in_grad[1], req[1], 0)
        self.assign(in_grad[2], req[2], 0)
        self.assign(in_grad[3], req[3], g_th)


@mx.operator.register("smoothed_focal_loss")
class SmoothedFocalLossProp(mx.operator.CustomOpProp):
    '''
    '''
    def __init__(self, alpha=0.25, gamma=2.0, th_prob=0.1, w_reg=1.0, normalize=False):
        #
        super(SmoothedFocalLossProp, self).__init__(need_top_grad=False)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.th_prob = float(th_prob)
        self.w_reg = float(w_reg)
        self.normalize = bool(literal_eval(str(normalize)))

    def list_arguments(self):
        return ['cls_pred', 'cls_prob', 'cls_target', 'th_prob']

    def list_outputs(self):
        return ['cls_prob']

    def infer_shape(self, in_shape):
        # in_shape[3] = (1,)
        out_shape = [in_shape[0], ]
        return in_shape, out_shape

    # def infer_type(self, in_type):
    #     dtype = in_type[0]
    #     import ipdb
    #     ipdb.set_trace()
    #     return [dtype, dtype, dtype, dtype], [dtype], []

    def create_operator(self, ctx, shapes, dtypes):
        return SmoothedFocalLoss(self.alpha, self.gamma, self.th_prob, self.w_reg, self.normalize)
