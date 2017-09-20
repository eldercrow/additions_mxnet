import mxnet as mx
import numpy as np
from config.config import cfg
import cv2


class MultiBoxMetric(mx.metric.EvalMetric):
    """Calculate metrics for Multibox training """
    def __init__(self, fn_stat=None, eps=1e-08):
        # managed internally, for debug only
        self.fn_stat = fn_stat
        self.aphw_grid = np.zeros((100, 100), dtype=np.int64)
        self.pphw_grid = np.zeros((100, 100), dtype=np.float64)
        self.eps = eps
        self.reset_stat = 0

        super(MultiBoxMetric, self).__init__('MultiBox')
        self.use_focal_loss = cfg.train['use_focal_loss']
        self.use_smooth_ce = cfg.train['use_smooth_ce']
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
        if self.fn_stat and self.reset_stat % 10 == 0:
            with open(self.fn_stat, 'w') as fh:
                # import ipdb
                # ipdb.set_trace()
                for l in self.aphw_grid:
                    lstr = '  '.join(('{:>8d}'.format(a) for a in l))
                    fh.write(lstr + '\n')
                for p, n in zip(self.pphw_grid, self.aphw_grid):
                    pp = p / np.maximum(n, 1)
                    lstr = '  '.join(('{:>1.3f}'.format(a) for a in pp))
                    fh.write(lstr + '\n')
            normalized_pp = self.pphw_grid / np.maximum(self.aphw_grid, 1)
            normalized_pp /= np.maximum(self.eps, np.max(normalized_pp))
            normalized_ap = self.aphw_grid / float(np.maximum(np.max(self.aphw_grid), 1))
            cv2.imwrite(self.fn_stat.replace('.txt', '_pphw.png'), (normalized_pp * 255).astype(int))
            cv2.imwrite(self.fn_stat.replace('.txt', '_aphw.png'), (normalized_ap * 255).astype(int))

            self.aphw_grid = np.zeros((100, 100), dtype=np.int64)
            self.pphw_grid = np.zeros((100, 100), dtype=np.float64)
        self.reset_stat += 1

    def update(self, labels, preds):
        """
        Implementation of updating metrics
        """
        # get generated multi label from network
        cls_prob = mx.nd.pick(preds[0], preds[2], axis=1, keepdims=True).asnumpy()
        # cls_prob = np.maximum(cls_prob, self.eps)
        cls_label = preds[2].asnumpy()
        loss = -np.log(np.maximum(cls_prob, self.eps))
        if self.use_focal_loss:
            gamma = float(cfg.train['focal_loss_gamma'])
            alpha = float(cfg.train['focal_loss_alpha'])
            if self.use_smooth_ce:
                w_reg = float(cfg.train['smooth_ce_lambda'])
                th_prob = preds[6].asnumpy()[0] #float(cfg.train['smooth_ce_th'])
                loss1 = -cls_prob / th_prob - np.log(th_prob) + 1
                idx = cls_prob < th_prob
                loss[idx] = loss1[idx]
                loss += th_prob * th_prob * w_reg
            loss *= np.power(1 - cls_prob, gamma)
            loss[cls_label > 0] *= alpha
            loss[cls_label ==0] *= 1 - alpha
        elif self.use_smooth_ce:
            th_prob = float(cfg.train['smooth_ce_th'])
            loss1 = -cls_prob / th_prob - np.log(th_prob) + 1
            idx = cls_prob < th_prob
            loss[idx] = loss1[idx]
        loss *= (cls_label >= 0)

        self.sum_metric[0] += loss.sum()
        self.num_inst[0] += np.sum(cls_label > 0) if self.use_focal_loss else np.sum(cls_label >= 0)

        # smoothl1loss
        loc_loss = preds[1].asnumpy()
        loc_label = preds[3].asnumpy()
        self.sum_metric[1] += np.sum(loc_loss)
        self.num_inst[1] += np.sum(loc_label)

        match_info = preds[5].asnumpy()

        n_batch = cls_prob.shape[0]
        cls_label = labels[0].asnumpy()
        for prob, label, minfo in zip(cls_prob, cls_label, match_info): # for each batch
            vidx = np.where(np.any(label != -1, axis=1))[0]
            label = label[vidx, :]
            minfo = minfo[vidx, :]

            wid = np.minimum(99, (label[:, 3] - label[:, 1]) * 100).astype(int)
            hid = np.minimum(99, (label[:, 4] - label[:, 2]) * 100).astype(int)
            if len(label) == 1:
                wid = [wid]
                hid = [hid]

            for w, h, m in zip(wid, hid, minfo):
                n_anc = np.sum(m >= 0)
                sum_prob = np.sum(prob[0][m.astype(int)] * (m >= 0))
                self.aphw_grid[h, w] += n_anc
                self.pphw_grid[h, w] += sum_prob

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
