import mxnet as mx
import numpy as np


class ReweightLoss(mx.operator.CustomOp):
    '''
    '''
    def __init__(self, neg_ratio, rand_mult):
        super(ReweightLoss, self).__init__()
        self.neg_ratio = neg_ratio
        self.rand_mult = rand_mult


    def forward(self, is_train, req, in_data, out_data, aux):
        '''
        '''
        cls_prob = in_data[0] # (n_batch, n_class, n_anchor)
        cls_target = in_data[1] # (n_batch, n_anchor)

        n_batch = cls_prob.shape[0]

        prob_gt = mx.nd.maximum(0.0, 1.0 - mx.nd.pick(cls_prob, cls_target, 1)) # (n_batch, n_anchor)
        # psq = mx.nd.power(prob_gt, self.rand_mult)

        pos_map = (cls_target > 0)
        n_pos = mx.nd.sum(pos_map).asscalar()
        n_neg = np.maximum(1, int(n_pos / n_batch * self.neg_ratio))

        neg_map = (cls_target == 0)
        neg_map = prob_gt * neg_map

        # pick some (more than needed) semi-hard negative samples, according to loss,
        # then randomly choose hard samples
        th_batch = np.transpose(mx.nd.topk(neg_map, axis=1, k=n_neg*self.rand_mult).asnumpy())
        np.random.shuffle(th_batch)
        th_batch = np.transpose(th_batch[:n_neg]).astype(np.int32)

        # assume ignore label is -1, num_classes is smaller than 1000
        # set ohem map as -1000 (that will be ignored) or 0
        # also make ohem map as 0 for positive samples
        nneg = np.full(neg_map.shape, -1000, dtype=np.float32)
        for i, th in enumerate(th_batch):
            nneg[i, th] = 0
        # 0 or -1000
        ohem_map = mx.nd.array(nneg, ctx=cls_target.context)
        ohem_map *= (1 - pos_map)
        ohem_map = mx.nd.maximum(-1, cls_target + ohem_map)

        self.assign(out_data[0], req[0], in_data[0])
        self.assign(out_data[1], req[1], ohem_map)


    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        '''
        '''
        # cls_prob = in_data[0] # (n_batch, n_class, n_anchor)
        # cls_target = in_data[1] # (n_batch, n_anchor)
        #
        # n_batch = cls_prob.shape[0]
        #
        # prob_gt = mx.nd.maximum(0.0, 1.0 - mx.nd.pick(cls_prob, cls_target, 1)) # (n_batch, n_anchor)
        # # psq = mx.nd.power(prob_gt, self.rand_mult)
        #
        # pos_map = (cls_target > 0)
        # n_pos = mx.nd.sum(pos_map).asscalar()
        # n_neg = np.maximum(1, int(n_pos / n_batch * self.neg_ratio))
        #
        # neg_map = (cls_target == 0)
        # neg_map = prob_gt * neg_map
        # th_batch = np.transpose(mx.nd.topk(neg_map, axis=1, k=n_neg*3).asnumpy())
        # np.random.shuffle(th_batch)
        # th_batch = np.transpose(th_batch[:n_neg]).astype(np.int32)
        # nneg = np.zeros(neg_map.shape, dtype=np.float32)
        # for i, th in enumerate(th_batch):
        #     nneg[i, th] = 1
        # neg_map = mx.nd.array(nneg, ctx=cls_prob.context)
        #
        # weight_map = pos_map + neg_map
        # weight_map = mx.nd.reshape(weight_map, shape=(0, 1, -1))

        weight_map = mx.nd.reshape(out_data[1] >= 0, shape=(0, 1, -1))

        self.assign(in_grad[0], req[0], out_grad[0] * weight_map)
        self.assign(in_grad[1], req[1], 0)


@mx.operator.register("reweight_loss")
class ReweightLossProp(mx.operator.CustomOpProp):
    '''
    '''
    def __init__(self, neg_ratio=3.0, rand_mult=8):
        #
        super(ReweightLossProp, self).__init__(need_top_grad=True)
        self.neg_ratio = float(neg_ratio)
        self.rand_mult = int(float(rand_mult))


    def list_arguments(self):
        return ['cls_prob', 'cls_target']


    def list_outputs(self):
        return ['cls_prob', 'cls_target']


    def infer_shape(self, in_shape):
        return in_shape, in_shape, []


    def create_operator(self, ctx, shapes, dtypes):
        return ReweightLoss(self.neg_ratio, self.rand_mult)
