import mxnet as mx
import numpy as np

from rcnn.config import config


def get_rpn_names():
    pred = ['w_rpn_cls_loss', 'w_rpn_bbox_loss', 'rpn_cls_prob']
    label = ['rpn_label', 'rpn_bbox_target', 'rpn_bbox_weight']
    return pred, label


def get_rcnn_names():
    pred = ['w_rcnn_cls_loss', 'w_rcnn_bbox_loss', 'rcnn_cls_prob', 'rcnn_label']
    rpn_pred, rpn_label = get_rpn_names()
    pred = rpn_pred + pred
    label = rpn_label
    return pred, label


class RPNAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNAccMetric, self).__init__('RPNAcc')
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rpn_cls_prob')]
        label = labels[self.label.index('rpn_label')]

        # pred (b, c, p) or (b, c, h, w)
        pred_label = mx.ndarray.argmax_channel(pred).asnumpy().astype('int32')
        pred_label = pred_label.reshape((pred_label.shape[0], -1))
        # label (b, p)
        label = label.asnumpy().astype('int32')

        # filter with keep_inds
        keep_inds = np.where(label != -1)
        pred_label = pred_label[keep_inds]
        label = label[keep_inds]

        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)


class RCNNAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RCNNAccMetric, self).__init__('RCNNAcc')
        self.pred, self.label = get_rcnn_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rcnn_cls_prob')]
        label = preds[self.pred.index('rcnn_label')]

        last_dim = pred.shape[-1]
        pred_label = pred.asnumpy().reshape(-1, last_dim).argmax(axis=1).astype('int32')
        label = label.asnumpy().reshape(-1,).astype('int32')

        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)


class RPNLossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNLossMetric, self).__init__('RPNLoss')
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('w_rpn_cls_loss')]
        label = labels[self.label.index('rpn_label')]

        # label (b, p)
        label = label.asnumpy().astype('int32')

        # filter with keep_inds
        keep_inds = np.where(label != -1)
        label = label[keep_inds]

        self.sum_metric += pred.asnumpy()[0]
        self.num_inst += len(label.flat)


class RCNNLossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RCNNLossMetric, self).__init__('RCNNLoss')
        # self.e2e = config.TRAIN.END2END
        self.pred, self.label = get_rcnn_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('w_rcnn_cls_loss')]
        label = preds[self.pred.index('rcnn_label')]

        # last_dim = pred.shape[-1]
        # pred_label = pred.asnumpy().reshape(-1, last_dim).argmax(axis=1).astype('int32')
        label = label.asnumpy().reshape(-1,).astype('int32')

        self.sum_metric += pred.asnumpy()[0]
        self.num_inst += len(label.flat)

        
class RPNL1LossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNL1LossMetric, self).__init__('RPNL1')
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):
        bbox_loss = preds[self.pred.index('w_rpn_bbox_loss')].asnumpy()
        bbox_weight = labels[self.label.index('rpn_bbox_weight')].asnumpy()

        # calculate num_inst (average on those fg anchors)
        num_inst = np.sum(bbox_weight > 0) / 4

        self.sum_metric += bbox_loss[0]
        self.num_inst += num_inst


class RCNNL1LossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RCNNL1LossMetric, self).__init__('RCNNL1')
        self.e2e = config.TRAIN.END2END
        self.pred, self.label = get_rcnn_names()

    def update(self, labels, preds):
        bbox_loss = preds[self.pred.index('w_rcnn_bbox_loss')].asnumpy()
        label = preds[self.pred.index('rcnn_label')].asnumpy()

        # calculate num_inst
        keep_inds = np.where(label != 0)[0]
        num_inst = len(keep_inds)

        self.sum_metric += bbox_loss[0]
        self.num_inst += num_inst
