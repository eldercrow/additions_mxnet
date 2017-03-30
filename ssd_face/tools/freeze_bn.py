import mxnet as mx
import numpy as np
import os, sys
sys.path.append(os.path.abspath('../symbol'))
from multibox_prior_layer import *
from anchor_target_layer import *

def freeze_bn(model_prefix, num_epoch, res_prefix):
    net, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, num_epoch)

    for k in aux_params:
        if k.endswith('_moving_var'):
            mm = k.replace('_moving_var', '_moving_mean')
            gamma = k.replace('_moving_var', '_gamma')
            beta = k.replace('_moving_var', '_beta')
            arg_params[gamma] /= aux_params[k]
            arg_params[beta] -= aux_params[mm] / aux_params[k]
            aux_params[k] = mx.nd.ones(aux_params[k].shape)
            aux_params[mm] = mx.nd.zeros(aux_params[mm].shape)

    mx.model.save_checkpoint(res_prefix, num_epoch, net, arg_params, aux_params)

if __name__ == '__main__':
    model_prefix = '/home/hyunjoon/github/additions_mxnet/ssd_face/model/pvtnet_preact_patch_256'
    num_epoch = 15
    res_prefix = model_prefix + '_bnfixed'

    freeze_bn(model_prefix, num_epoch, res_prefix)

    _, arg_params, aux_params = mx.model.load_checkpoint(res_prefix, num_epoch)
