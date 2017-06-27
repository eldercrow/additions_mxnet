import mxnet as mx
import numpy as np


class MultiBoxTarget(mx.operator.CustomOp):
    '''
    '''
    def __init__(self):
        super(MultiBoxTarget, self).__init__()


@mx.operator.register("multibox_target")
class MultiBoxTargetProp(mx.operator.CustomOpProp):
    '''
    '''
    def __init__(self, neg_ratio=3.0, exp_loss=2.0, ignore_label=-1):
        #
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
        return MultiBoxTarget(self.neg_ratio, self.exp_loss, self.ignore_label)
