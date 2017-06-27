import mxnet as mx
import numpy as np


class ReweightLoss(mx.operator.CustomOp):
    '''
    '''
    def __init__(self, neg_ratio, exp_loss, ignore_label):
        super(ReweightLoss, self).__init__()
        self.neg_ratio = neg_ratio
        self.exp_loss = exp_loss
        self.ignore_label = ignore_label


    def forward(self, is_train, req, in_data, out_data, aux):
        '''
        '''
        self.assign(out_data[0], req[0], in_data[0])


    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        '''
        '''
        cls_prob = in_data[0] # (n_batch, n_class, n_anchor)
        label_map = in_data[1] # (n_batch, n_anchor)

        prob_gt = 1 - mx.nd.pick(cls_prob, label_map, 1) # (n_batch, n_anchor)
        psq = mx.nd.power(prob_gt, self.exp_loss)

        pos_map = psq * (label_map > 0)
        max_pos = mx.nd.max(pos_map).asscalar()
        pos_map /= max_pos

        neg_map = psq * (label_map == 0)
        max_neg = mx.nd.max(neg_map).asscalar()
        neg_map /= max_neg

        sum_pos = mx.nd.sum(pos_map).asscalar()
        sum_neg = mx.nd.sum(neg_map).asscalar()
        ratio_neg = sum_pos * self.neg_ratio / sum_neg

        weight_map = pos_map + neg_map * ratio_neg
        weight_map = mx.nd.reshape(weight_map, shape=(n_batch, 1, -1))
        self.assign(in_grad[0], req[0], mx.nd.tile(weight_map, (1, in_data[0].shape[1], 1)))


@mx.operator.register("reweight_loss")
class ReweightLossProp(mx.operator.CustomOpProp):
    '''
    '''
    def __init__(self, neg_ratio=3.0, exp_loss=2.0, ignore_label=-1):
        #
        super(ReweightLossProp, self).__init__(need_top_grad=True)
        self.neg_ratio = float(neg_ratio)
        self.exp_loss = float(exp_loss)
        self.ignore_label = int(ignore_label)


    def list_arguments(self):
        return ['cls_prob', 'label_map']


    def list_outputs(self):
        return ['cls_prob']


    def infer_shape(self, in_shape):
        return in_shape, [in_shape[0]], []


    def create_operator(self, ctx, shapes, dtypes):
        return ReweightLoss(self.neg_ratio, self.exp_loss, self.ignore_label)
