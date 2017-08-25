import mxnet as mx
import numpy as np
from config.config import cfg


class MultiBoxMetric(mx.metric.EvalMetric):
    """Calculate metrics for Multibox training """
    def __init__(self, eps=1e-8):
        super(MultiBoxMetric, self).__init__('MultiBox')
        self.eps = eps
        self.use_focal_loss = cfg.train['use_focal_loss']
        self.num = 2
        self.name = ['CrossEntropy', 'SmoothL1']
        self.reset()

        # managed internally, for debug only
        self.aphw_grid = np.zeros((100, 100), dtype=np.int64)
        self.pphw_grid = np.zeros((100, 100), dtype=np.float64)

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
        match_info = preds[5].asnumpy().astype(int)

        import ipdb
        ipdb.set_trace()

        vc_cls = np.sum(cls_label >= 0)
        if self.use_focal_loss:
            vc_cls = np.sum(cls_label > 0)
        vc_loc = np.sum(loc_label)

        # overall accuracy & object accuracy
        label = cls_label.flatten()
        mask = np.where(label >= 0)[0]
        indices = np.int64(label[mask])
        prob = cls_prob.transpose((0, 2, 1)).reshape((-1, cls_prob.shape[1]))
        prob = prob[mask, indices]
        loss = -np.log(prob + self.eps)
        if self.use_focal_loss:
            label = label[mask]
            gamma = float(cfg.train['focal_loss_gamma'])
            alpha = float(cfg.train['focal_loss_alpha'])
            loss *= np.power(1 - prob, gamma)
            loss[label > 0] *= alpha
            loss[label ==0] *= 1 - alpha
        self.sum_metric[0] += loss.sum()
        self.num_inst[0] += vc_cls
        # smoothl1loss
        self.sum_metric[1] += np.sum(loc_loss)
        self.num_inst[1] += vc_loc

        n_batch = cls_prob.shape[0]
        cls_prob = np.reshape(prob, (n_batch, -1))
        for prob, label, minfo in zip(cls_prob, cls_label, match_info): # for each batch
            vidx = np.where(np.any(label, axis=1))[0]
            label = label[vidx, :]
            minfo = minfo[vidx, :]
            wid = np.minimum(99, int((label[:, 3] - label[:, 1]) * 100))
            hid = np.minimum(99, int((label[:, 4] - label[:, 2]) * 100))

            for w, h, m in zip(wid, hid, minfo):
                self.aphw_grid[h, w] += np.sum(m >= 0)
                self.pphw_grid[h, w] += np.sum(prob[minfo] * m >= 0)

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
