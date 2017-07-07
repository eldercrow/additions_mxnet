import mxnet as mx
import numpy as np


class ReweightLoss(mx.operator.CustomOp):
    '''
    '''
    def __init__(self, th_iou, th_iou_neg, neg_ratio, rand_mult):
        super(ReweightLoss, self).__init__()
        self.th_iou = th_iou
        self.th_iou_neg = th_iou_neg
        self.neg_ratio = neg_ratio
        self.rand_mult = rand_mult


    def forward(self, is_train, req, in_data, out_data, aux):
        '''
        '''
        cls_prob = in_data[0] # (n_batch, n_class, n_anchor)
        cls_target = in_data[1] # (n_batch, n_anchor)
        bbox_weight = in_data[2]
        max_iou = in_data[3]

        n_batch = cls_prob.shape[0]

        # regression sample masking
        pos_mask = max_iou > self.th_iou
        reg_mask = 1 - pos_mask
        reg_mask *= max_iou > self.th_iou_neg

        bbox_mask = reg_mask * (mx.nd.argmax(cls_prob, axis=1) == cls_target)
        bbox_mask += pos_mask
        bbox_weight = mx.nd.broadcast_mul(in_data[2], mx.nd.reshape(bbox_mask, (0, 0, 1)))

        prob_gt = mx.nd.maximum(0.0, 1.0 - mx.nd.pick(cls_prob, cls_target, 1)) # (n_batch, n_anchor)

        pos_map = (cls_target > 0) * pos_mask
        n_pos = mx.nd.sum(pos_map).asscalar()
        n_neg = np.maximum(1, int(n_pos / n_batch * self.neg_ratio))

        # negative sample: random sampling + hard negative mining
        neg_map = prob_gt * (cls_target == 0)

        # pick some (more than needed) semi-hard negative samples, according to loss,
        # then randomly choose hard samples
        th_batch = mx.nd.topk(neg_map, axis=1, k=n_neg*self.rand_mult).asnumpy()
        for i in range(th_batch.shape[0]):
          np.random.shuffle(th_batch[i])
        th_batch = th_batch[:, :n_neg].astype(np.int32)

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
        self.assign(out_data[2], req[2], bbox_weight)
        self.assign(out_data[3], req[3], mx.nd.maximum(np.finfo(np.float32).eps, 4.0 / mx.nd.sum(bbox_weight)))
        # self.assign(out_data[0], req[0], ohem_map)
        # self.assign(out_data[1], req[1], bbox_weight)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        '''
        '''
        weight_map = mx.nd.reshape(out_data[1] >= 0, shape=(0, 1, -1))
        sum_weight = mx.nd.sum(weight_map) # 'valid'
        weight_map /= sum_weight

        self.assign(in_grad[0], req[0], out_grad[0] * weight_map)
        for i in range(1, 4):
            self.assign(in_grad[i], req[i], 0)


@mx.operator.register("reweight_loss")
class ReweightLossProp(mx.operator.CustomOpProp):
    '''
    '''
    def __init__(self, th_iou=0.5, th_iou_neg=1.0 / 3.0, neg_ratio=3.0, rand_mult=3):
        #
        super(ReweightLossProp, self).__init__(need_top_grad=True)
        self.th_iou = float(th_iou)
        self.th_iou_neg = float(th_iou_neg)
        self.neg_ratio = float(neg_ratio)
        self.rand_mult = int(float(rand_mult))


    def list_arguments(self):
        return ['cls_prob', 'cls_target', 'bbox_weight', 'max_iou']


    def list_outputs(self):
        # return ['cls_target', 'bbox_weight']
        return ['cls_prob', 'cls_target', 'bbox_weight', 'bbox_norm']


    def infer_shape(self, in_shape):
        # return in_shape, in_shape[1:-1], []
        out_shape = in_shape[:-1]
        out_shape.append((1,))
        return in_shape, out_shape, []


    def create_operator(self, ctx, shapes, dtypes):
        return ReweightLoss(self.th_iou, self.th_iou_neg, self.neg_ratio, self.rand_mult)
