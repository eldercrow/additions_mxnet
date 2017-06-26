import mxnet as mx
import numpy as np


class FacePatchMetric(mx.metric.EvalMetric):
    """Calculate metrics for Multibox training """
    def __init__(self, num=3, postfix_names=['Loss', 'SmoothL1', 'Recall']): #'LossOrig']):
        # super(FacePatchMetric, self).__init__(['Loss', 'SmoothL1', 'Recall'], 3)
        self.sum_metric = np.zeros((num,))
        self.num_inst = np.zeros((num,), dtype=int)
        super(FacePatchMetric, self).__init__(name='')
        self.postfix_names = postfix_names
        self.num = num

    def update(self, labels, preds):
        """
        Implementation of updating metrics
        labels are not used here.
        """
        # get generated multi label from network
        cls_loss = preds[0].asnumpy()
        reg_loss = preds[1].asnumpy()
        cls_label = preds[2].asnumpy()
        reg_label = preds[3].asnumpy()
        cls_prob = preds[4].asnumpy()

        mask = np.where(cls_label >= 0)[0]
        n_valid_sample = mask.size
        if mask.size > 0:
            self.sum_metric[0] += sum(cls_loss)

        mask = np.where(np.any(reg_label != -1, axis=1))[0]
        n_valid_sample += mask.size
        if mask.size > 0:
            self.sum_metric[1] += sum(reg_loss)
        self.num_inst[0] += n_valid_sample
        self.num_inst[1] += n_valid_sample

        cls_acc = np.argmax(cls_prob, axis=1) == cls_label
        mask = np.where(cls_label > 0)[0]
        if mask.size > 0:
            self.sum_metric[2] += sum(cls_acc[mask])
            self.num_inst[2] += mask.size

        # mask = np.where(cls_label > 0)[0]
        # cls_acc = np.argmax(cls_pred, axis=1) == cls_label
        # masked_cls_label = cls_label[mask].astype(int)
        # cls_loss = -np.log(np.maximum(cls_pred[mask, masked_cls_label], 1e-08))
        # if mask.size > 0:
        #     # self.sum_metric[0] += sum(cls_acc[mask])
        #     self.sum_metric[0] += sum(cls_loss)
        #     self.num_inst[0] += mask.size
        #
        # mask = np.where(cls_label > 0)[0]
        # if mask.size > 0:
        #     self.sum_metric[2] += sum(cls_acc[mask])
        #     self.num_inst[2] += mask.size
        #
        # reg_dist = np.sum(np.abs(reg_pred - reg_label), axis=1)
        # # import ipdb
        # # ipdb.set_trace()
        # if mask.size > 0:
        #     self.sum_metric[1] += sum(reg_dist[mask])
        #     self.num_inst[1] += mask.size

    def update_dict(self, labels, preds):
        label = []
        pred = []
        for l, v in labels.items():
            label.append(v)
        for p, v in preds.items():
            pred.append(v)

        self.update(label, pred)

    def reset(self):
        self.sum_metric[:] = 0.0
        self.num_inst[:] = 0

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
        # if self.num is None:
        #     if self.num_inst == 0:
        #         return (self.name, float('nan'))
        #     else:
        #         return (self.name, self.sum_metric / self.num_inst)
        # else:
        names = ['%s'%(self.name+self.postfix_names[i]) for i in range(self.num)]
        values = [x / y if y != 0 else float('nan') \
            for x, y in zip(self.sum_metric, self.num_inst)]
        return (names, values)

# class MultiBoxMetric(mx.metric.EvalMetric):
#     """Calculate metrics for Multibox training """
#     def __init__(self):
#         super(MultiBoxMetric, self).__init__(['Acc', 'ObjectAcc', 'SmoothL1'], 3)
#
#     def update(self, labels, preds):
#         """
#         Implementation of updating metrics
#         """
#         # get generated multi label from network
#         cls_prob = preds[0].asnumpy()
#         loc_loss = preds[1].asnumpy()
#         cls_label = preds[2].asnumpy()
#         # overall accuracy & object accuracy
#         label = cls_label.flatten()
#         mask = np.where(cls_label >= 0)[0]
#         # p = np.argmax(cls_prob, axis=2).flatten() # for python multibox_target
#         p = np.argmax(cls_prob, axis=1).flatten()
#         self.sum_metric[0] += np.sum(p[mask] == label[mask])
#         self.num_inst[0] += mask.size
#         mask = np.where(cls_label > 0)[0]
#         self.sum_metric[1] += np.sum(p[mask] == label[mask])
#         self.num_inst[1] += mask.size
#         # smoothl1loss
#         # self.sum_metric[2] += np.sum(loc_loss)
#         # self.num_inst[2] += loc_loss.shape[0]
#         self.sum_metric[2] += np.sum(loc_loss)
#         self.num_inst[2] += loc_loss.shape[0]
#
#     def get(self):
#         """Get the current evaluation result.
#         Override the default behavior
#
#         Returns
#         -------
#         name : str
#            Name of the metric.
#         value : float
#            Value of the evaluation.
#         """
#         if self.num is None:
#             if self.num_inst == 0:
#                 return (self.name, float('nan'))
#             else:
#                 return (self.name, self.sum_metric / self.num_inst)
#         else:
#             names = ['%s'%(self.name[i]) for i in range(self.num)]
#             values = [x / y if y != 0 else float('nan') \
#                 for x, y in zip(self.sum_metric, self.num_inst)]
#             return (names, values)
#
# class FaceMetric(mx.metric.EvalMetric):
#     """Calculate metrics for Multibox training """
#     def __init__(self):
#         super(FaceMetric, self).__init__(['ObjectAcc', 'SmoothL1'], 2)
#
#     def update(self, labels, preds):
#         """
#         Implementation of updating metrics
#         labels are not used here.
#         """
#         # get generated multi label from network
#         cls_pred = preds[0].asnumpy()
#         reg_pred = preds[1].asnumpy()
#         cls_label = preds[2].asnumpy()
#         reg_label = preds[3].asnumpy()
#
#         cls_diff = np.sum((cls_pred - cls_label)**2, axis=1)
#         mask = np.where(cls_label[:, 0] != -1)[0]
#         if mask.size > 0:
#             self.sum_metric[0] += sum(cls_diff[mask])
#             self.num_inst[0] += mask.size
#
#         mask = np.where(cls_label[:, 0] == 0)[0]
#         reg_dist = np.sum(np.abs(reg_pred - reg_label), axis=1)
#         # if any(reg_dist > 1000000000):
#         #     import ipdb
#         #     ipdb.set_trace()
#         if mask.size > 0:
#             self.sum_metric[1] += sum(reg_dist[mask])
#             self.num_inst[1] += mask.size
#
#     def get(self):
#         """Get the current evaluation result.
#         Override the default behavior
#
#         Returns
#         -------
#         name : str
#            Name of the metric.
#         value : float
#            Value of the evaluation.
#         """
#         if self.num is None:
#             if self.num_inst == 0:
#                 return (self.name, float('nan'))
#             else:
#                 return (self.name, self.sum_metric / self.num_inst)
#         else:
#             names = ['%s'%(self.name[i]) for i in range(self.num)]
#             values = [x / y if y != 0 else float('nan') \
#                 for x, y in zip(self.sum_metric, self.num_inst)]
#             return (names, values)

