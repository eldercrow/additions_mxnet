import mxnet as mx
import numpy as np


class MultiBoxMetric(mx.metric.EvalMetric):
    """Calculate metrics for Multibox training """
    def __init__(self, eps=1e-8, use_focal_loss=False):
        super(MultiBoxMetric, self).__init__('MultiBox')
        self.eps = eps
        self.use_focal_loss = use_focal_loss
        self.num = 2
        self.name = ['CrossEntropy', 'SmoothL1']
        self.reset()

    def reset(self):
        """
        override reset behavior
        """
        if getattr(self, 'num', None) is None:
            self.num_inst = 0
            self.sum_metric = 0.0
        else:
            self.num_inst = [0] * self.num
            self.sum_metric = [0.0] * self.num

    def update(self, labels, preds):
        """
        Implementation of updating metrics
        """
        # get generated multi label from network
        cls_prob = preds[0].asnumpy()
        loc_loss = preds[1].asnumpy()
        cls_label = preds[2].asnumpy()
        loc_label = preds[3].asnumpy()
        valid_count = np.sum(cls_label >= 0)
        if self.use_focal_loss:
            valid_count += np.sum(loc_label)
            valid_count /= 100.0 # just for better display
            vc_cls, vc_loc = valid_count, valid_count
        else:
            vc0 = valid_count
            vc1 = np.sum(cls_label > 0, axis=2)
            vc2 = np.sum(vc1 == 0)
            vc1 = np.sum(vc1)
            vc_cls = np.minimum(vc0, vc1 * 4.0 + vc2)
            # vc_cls = valid_count
            vc_loc = np.sum(loc_label)
        # overall accuracy & object accuracy
        label = cls_label.flatten()
        mask = np.where(label >= 0)[0]
        indices = np.int64(label[mask])
        prob = cls_prob.transpose((0, 2, 1)).reshape((-1, cls_prob.shape[1]))
        prob = prob[mask, indices]
        loss = -np.log(prob + self.eps)
        if self.use_focal_loss:
            loss *= np.power(1 - prob, 5.0) * 0.1
        self.sum_metric[0] += loss.sum()
        self.num_inst[0] += vc_cls
        # smoothl1loss
        self.sum_metric[1] += np.sum(loc_loss)
        self.num_inst[1] += vc_loc

    def get(self):
        """Get the current evaluation result.
        Override the default behavior

        Returns
        -------
        name : str
           Name of the metric.
        value : float
           Value of the evaluation.
        """
        if self.num is None:
            if self.num_inst == 0:
                return (self.name, float('nan'))
            else:
                return (self.name, self.sum_metric / self.num_inst)
        else:
            names = ['%s'%(self.name[i]) for i in range(self.num)]
            values = [x / y if y != 0 else float('nan') \
                for x, y in zip(self.sum_metric, self.num_inst)]
            return (names, values)
