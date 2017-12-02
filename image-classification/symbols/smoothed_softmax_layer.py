import mxnet as mx
import numpy as np
# from ast import literal_eval
import logging


class SmoothedSoftmaxLoss(mx.operator.CustomOp):
    '''
    '''
    def __init__(self, w_reg, normalization):
        super(SmoothedSoftmaxLoss, self).__init__()
        self.w_reg = w_reg
        self.normalization = normalization
        self.eps = np.finfo(np.float32).eps

    def forward(self, is_train, req, in_data, out_data, aux):
        '''
        Just pass the data.
        '''
        self.assign(out_data[0], req[0], in_data[1])
        th_prob = in_data[3].asscalar() / self.w_reg

        p = mx.nd.pick(in_data[1], in_data[2], axis=1, keepdims=False)
        ce = -mx.nd.log(mx.nd.maximum(p, self.eps))
        sce = -p / th_prob - np.log(th_prob) + 1

        mask = p > th_prob
        sce = mask * ce + (1 - mask) * sce # smoothed cross entropy]

        ce *= in_data[2] >= 0
        sce *= in_data[2] >= 0

        sce += th_prob #* self.w_reg * self.w_reg
        self.assign(out_data[1], req[1], sce)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        '''
        Reweight loss according to focal loss.
        '''
        n_class = in_data[1].shape[1]
        th_prob = in_data[3].asscalar() / self.w_reg

        p = mx.nd.pick(in_data[1], in_data[2], axis=1, keepdims=True)

        pmask = p > th_prob

        alpha = mx.nd.one_hot(mx.nd.reshape(in_data[2], (0, -1)), n_class,
                on_value=1, off_value=0)
        alpha = mx.nd.reshape(alpha, (0, -1))

        # ordinary cross entropy
        g0 = in_data[1] - alpha

        # p < th_prob
        g1 = g0 * p / th_prob

        # combined
        g = g0 * pmask + g1 * (1 - pmask)
        g *= mx.nd.reshape((in_data[2] >= 0), (-1, 1))

        g_th = mx.nd.minimum(p / th_prob, 1.0) / th_prob - 1.0 / th_prob
        g_th /= self.w_reg
        g_th = mx.nd.sum(g_th) + in_data[2].size * th_prob * 2.0 #* self.w_reg

        if self.normalization == 'valid':
            norm = mx.nd.sum(in_data[2] >= 0).asscalar()
            g /= norm
            g_th /= norm
        elif self.normalization == 'batch':
            g /= in_data[2].size
            g_th /= in_data[2].size

        if mx.nd.uniform(0, 1, (1,)).asscalar() < 0.001:
            logging.getLogger().info('Current th_prob for smoothed CE: {}'.format(th_prob))

        self.assign(in_grad[0], req[0], g)
        self.assign(in_grad[1], req[1], 0)
        self.assign(in_grad[2], req[2], 0)
        self.assign(in_grad[3], req[3], g_th)


@mx.operator.register("smoothed_softmax_loss")
class SmoothedSoftmaxLossProp(mx.operator.CustomOpProp):
    '''
    '''
    def __init__(self, w_reg=1.0, normalization='none'):
        #
        super(SmoothedSoftmaxLossProp, self).__init__(need_top_grad=False)
        self.w_reg = float(w_reg)
        self.normalization = normalization

    def list_arguments(self):
        return ['cls_pred', 'cls_prob', 'cls_target', 'th_prob']

    def list_outputs(self):
        # return ['cls_target', 'bbox_weight']
        return ['cls_prob', 'cls_loss']

    def infer_shape(self, in_shape):
        out_shape = [in_shape[0], in_shape[2]]
        return in_shape, out_shape, []

    def create_operator(self, ctx, shapes, dtypes):
        return SmoothedSoftmaxLoss(self.w_reg, self.normalization)
