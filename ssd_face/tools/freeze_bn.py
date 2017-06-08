import mxnet as mx
import numpy as np
import os, sys
sys.path.append(os.path.abspath('../layer'))
from multibox_prior_layer import *
from multibox_target import *
from softmax_loss import *

def freeze_bn(model_prefix, num_epoch, res_prefix):
    net, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, num_epoch)

    for k in aux_params:
        if k.endswith('_moving_var'):
            mm = k.replace('_moving_var', '_moving_mean')
            gamma = k.replace('_moving_var', '_gamma')
            beta = k.replace('_moving_var', '_beta')
            if gamma in arg_params and beta in arg_params:
                print 'converting...'
                sigma = mx.nd.sqrt(aux_params[k] + 1e-05)
                arg_params[gamma] /= sigma
                arg_params[beta] -= aux_params[mm] * arg_params[gamma]
            aux_params[k] = mx.nd.ones(aux_params[k].shape) - 1e-05
            aux_params[mm] = mx.nd.zeros(aux_params[mm].shape)

    mx.model.save_checkpoint(res_prefix, num_epoch, net, arg_params, aux_params)

if __name__ == '__main__':
    model_prefix = '/home/hyunjoon/github/additions_mxnet/ssd_face/model/spotnet_lighter_trainval_768'
    num_epoch = 35
    res_prefix = model_prefix.replace('_768', '') + '_bnfixed_768'

    freeze_bn(model_prefix, num_epoch, res_prefix)

    # _, arg_params, aux_params = mx.model.load_checkpoint(res_prefix, num_epoch)
    # import ipdb
    # ipdb.set_trace()
