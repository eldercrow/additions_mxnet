import mxnet as mx
import numpy as np
import os, sys
sys.path.append(os.path.abspath('../layer'))
from multibox_prior_layer import *
from multibox_target import *
from softmax_loss import *

def freeze_bn(model_prefix, num_epoch, res_prefix, res_epoch, clone_only=True):
    net, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, num_epoch)

    cloned_name_g = ['g3/u0', 'g4/u0', 'g5/u0', 'g6/u0']
    cloned_name_p = ['g3/proj', 'g4/proj', 'g5/proj', 'g6/proj']
    cloned_name_h = ['hyper096', 'hyper192', 'hyper384', 'hyper768']

    for k in sorted(aux_params):
        if clone_only:
            is_clone = False
            if k.find('clone/') >= 0:
                is_clone = True
            for cn in cloned_name_h:
                if k.find(cn) >= 0:
                    is_clone = True
            if not is_clone:
                continue
        if k.endswith('_moving_var'):
            print 'Converting {}...'.format(k)
            mm = k.replace('_moving_var', '_moving_mean')
            if clone_only:
                for cn in cloned_name_g[1:]:
                    if k.find(cn) >= 0:
                        k0 = k.replace(cn, cloned_name_g[0])
                        mm0 = mm.replace(cn, cloned_name_g[0])
                        aux_params[k] = aux_params[k0].copy()
                        aux_params[mm] = aux_params[mm0].copy()
                for cn in cloned_name_h[1:]:
                    if k.find(cn) >= 0:
                        k0 = k.replace(cn, cloned_name_h[0])
                        mm0 = mm.replace(cn, cloned_name_h[0])
                        aux_params[k] = aux_params[k0].copy()
                        aux_params[mm] = aux_params[mm0].copy()
                for cn in cloned_name_p[1:]:
                    if k.find(cn) >= 0:
                        k0 = k.replace(cn, cloned_name_p[0])
                        mm0 = mm.replace(cn, cloned_name_p[0])
                        aux_params[k] = aux_params[k0].copy()
                        aux_params[mm] = aux_params[mm0].copy()
            else:
                gamma = k.replace('_moving_var', '_gamma')
                beta = k.replace('_moving_var', '_beta')
                if gamma in arg_params and beta in arg_params:
                    print 'Converting {}'.format(k)
                    sigma = mx.nd.sqrt(aux_params[k] + 1e-05)
                    arg_params[gamma] /= sigma
                    arg_params[beta] -= aux_params[mm] * arg_params[gamma]
                else:
                    print 'Merging {}'.format(k)
                aux_params[k] = mx.nd.ones(aux_params[k].shape) - 1e-05
                aux_params[mm] = mx.nd.zeros(aux_params[mm].shape)

    # import ipdb
    # ipdb.set_trace()
    mx.model.save_checkpoint(res_prefix, res_epoch, net, arg_params, aux_params)

if __name__ == '__main__':
    model_prefix = '/home/hyunjoon/github/additions_mxnet/ssd_face/model/spotnet_lighter3_768'
    num_epoch = 1000
    res_epoch = 0
    res_prefix = model_prefix.replace('_768', '') + '_bnfixed_768'

    freeze_bn(model_prefix, num_epoch, res_prefix, res_epoch, clone_only=True)

    # _, arg_params, aux_params = mx.model.load_checkpoint(res_prefix, num_epoch)
    # import ipdb
    # ipdb.set_trace()
