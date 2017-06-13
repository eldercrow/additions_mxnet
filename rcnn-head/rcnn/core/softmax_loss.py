import mxnet as mx
import numpy as np
from ast import literal_eval


class SoftmaxLoss(mx.operator.CustomOp):
    '''
    This layer simply computes softmax loss.
    No backward (gradient compute) is defined, 
    since it will be computed in SoftmaxOutput layer.
    '''
    def __init__(self, ignore_label=-1, use_ignore=False, 
            multi_output=False, preserve_shape=False, normalization='null'):
        self.ignore_label = ignore_label
        self.use_ignore = use_ignore
        self.multi_output = multi_output
        self.preserve_shape = preserve_shape
        self.normalization = normalization

    def forward(self, is_train, req, in_data, out_data, aux):
        '''
        Simply use CPU numpy for this layer.
        '''
        data = in_data[0].asnumpy()

        label = in_data[1].asnumpy().ravel().astype(int)
        if self.multi_output:
            n_class = data.shape[1]
            data = np.reshape(data, (data.shape[0], n_class, -1))
            data = np.reshape(np.transpose(data, (0, 2, 1)), (-1, n_class))
        if self.preserve_shape:
            data = np.reshape(data, (-1, data.shape[-1]))
        else:
            data = np.reshape(data, (data.shape[0], -1))

        is_good = np.all(np.isfinite(data), axis=1)
        if self.use_ignore == True:
            is_good = np.logical_and(is_good, label != self.ignore_label)
        vidx = np.where(is_good)[0]

        loss = 0.0 if vidx.size == 0 else -np.log(np.maximum(1e-08, data[vidx, label[vidx]]))
        if self.normalization == 'null':
            loss = np.sum(loss)
        elif self.normalization == 'valid':
            loss = np.mean(loss)
        elif self.normalization == 'batch':
            loss = np.sum(loss) / np.maximum(1, label.size)

        # for DEBUG
        self.assign(out_data[0], req[0], mx.nd.array((loss,), ctx=in_data[0].context))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        '''
        Reshape out_grad to in_grad.
        '''
        # if not self.multi_output:
        #     import ipdb
        #     ipdb.set_trace()
        self.assign(in_grad[0], req[0], mx.nd.tile(out_grad[0], in_grad[0].shape))


@mx.operator.register("softmax_loss")
class SoftmaxLossProp(mx.operator.CustomOpProp):
    def __init__(self, ignore_label=-1, use_ignore=False, 
            multi_output=False, preserve_shape=False, normalization='null'):
        #
        super(SoftmaxLossProp, self).__init__(need_top_grad=True)
        self.ignore_label = int(ignore_label)
        self.use_ignore = bool(literal_eval(str(use_ignore)))
        self.multi_output = bool(literal_eval(str(multi_output)))
        self.preserve_shape = bool(literal_eval(str(preserve_shape)))
        self.normalization = normalization

    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['output']

    def list_auxiliary_states(self):
        return []

    def infer_shape(self, in_shape):
        if not self.multi_output: 
            assert len(in_shape[0]) <= 3
            if self.preserve_shape == True:
                assert np.prod(in_shape[0]) / in_shape[0][-1] == np.prod(in_shape[1])
            else:
                assert in_shape[0][0] == np.prod(in_shape[1])
        else:
            assert np.prod(in_shape[0]) / in_shape[0][1] == np.prod(in_shape[1])

        out_shape = [(1,)]

        return in_shape, out_shape, []

    def create_operator(self, ctx, shapes, dtypes):
        return SoftmaxLoss(self.ignore_label, self.use_ignore, 
                self.multi_output, self.preserve_shape, self.normalization)
        
