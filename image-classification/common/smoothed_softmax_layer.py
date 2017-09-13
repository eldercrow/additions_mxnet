import mxnet as mx
import numpy as np
# from ast import literal_eval


class SmoothedSoftmaxLoss(mx.operator.CustomOp):
    '''
    '''
    def __init__(self, th_prob, normalization):
        super(SmoothedSoftmaxLoss, self).__init__()
        self.th_prob = th_prob
        self.normalization = normalization

    def forward(self, is_train, req, in_data, out_data, aux):
        '''
        Just pass the data.
        '''
        self.assign(out_data[0], req[0], in_data[1])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        '''
        Reweight loss according to focal loss.
        '''
        n_class = in_data[1].shape[1]
        p = mx.nd.pick(in_data[1], in_data[2], axis=1, keepdims=True)
        pmask = p > self.th_prob

        alpha = mx.nd.one_hot(in_data[2], n_class, on_value=1, off_value=0)
        # alpha = mx.nd.transpose(alpha, (0, 2, 1))

        # ordinary cross entropy
        g0 = in_data[1] - alpha

        # p < th_prob
        g1 = g0 * p / self.th_prob

        # combined
        valid_mask = mx.nd.reshape(in_data[2] >= 0, (-1, 1))
        g = g0 * pmask + g1 * (1 - pmask)
        g *= valid_mask

        if self.normalization == 'valid':
            g /= mx.nd.sum(valid_mask).asscalar()
        elif self.normalization == 'batch':
            g /= in_data[2].shape[0]

        self.assign(in_grad[0], req[0], g)
        self.assign(in_grad[1], req[1], 0)
        self.assign(in_grad[2], req[2], 0)


@mx.operator.register("smoothed_softmax_loss")
class SmoothedSoftmaxLossProp(mx.operator.CustomOpProp):
    '''
    '''
    def __init__(self, th_prob=0.001, normalization='valid'):
        #
        super(SmoothedSoftmaxLossProp, self).__init__(need_top_grad=False)
        self.th_prob = float(th_prob)
        self.normalization = normalization

    def list_arguments(self):
        return ['cls_pred', 'cls_prob', 'cls_target']

    def list_outputs(self):
        # return ['cls_target', 'bbox_weight']
        return ['cls_prob']

    def infer_shape(self, in_shape):
        out_shape = [in_shape[0], ]
        return in_shape, out_shape, []

    def create_operator(self, ctx, shapes, dtypes):
        return SmoothedSoftmaxLoss(self.th_prob, self.normalization)
