import mxnet as mx
import numpy as np


class SmoothedCrossEntropy(mx.metric.EvalMetric):
    """Computes Cross Entropy loss.

    The cross entropy over a batch of sample size :math:`N` is given by

    .. math::
       -\\sum_{n=1}^{N}\\sum_{k=1}^{K}t_{nk}\\log (y_{nk}),

    where :math:`t_{nk}=1` if and only if sample :math:`n` belongs to class :math:`k`.
    :math:`y_{nk}` denotes the probability of sample :math:`n` belonging to
    class :math:`k`.

    Parameters
    ----------
    eps : float
        Cross Entropy loss is undefined for predicted value is 0 or 1,
        so predicted values are added with the small constant.
    name : str
        Name of this metric instance for display.
    output_names : list of str, or None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.

    Examples
    --------
    >>> predicts = [mx.nd.array([[0.3, 0.7], [0, 1.], [0.4, 0.6]])]
    >>> labels   = [mx.nd.array([0, 1, 1])]
    >>> ce = mx.metric.CrossEntropy()
    >>> ce.update(labels, predicts)
    >>> print ce.get()
    ('cross-entropy', 0.57159948348999023)
    """
    def __init__(self, th_prob=1e-06, eps=1e-12, name='smoothed-cross-entropy',
                 output_names=None, label_names=None):
        super(SmoothedCrossEntropy, self).__init__(
            name, th_prob=th_prob, eps=eps,
            output_names=output_names, label_names=label_names)
        self.th_prob = th_prob
        self.eps = eps

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data.

        preds : list of `NDArray`
            Predicted values.
        """
        for label, pred in zip(labels, preds):
            label = label.asnumpy()
            pred = pred.asnumpy()

            label = label.ravel()
            assert label.shape[0] == pred.shape[0]

            prob = pred[np.arange(label.shape[0]), np.int64(label)]
            loss = -np.log(prob + self.eps)
            loss1 = -prob / self.th_prob - np.log(self.th_prob) + 1
            idx = prob < self.th_prob
            loss[idx] = loss1[idx]
            self.sum_metric += np.sum(loss)
            self.num_inst += label.shape[0]
