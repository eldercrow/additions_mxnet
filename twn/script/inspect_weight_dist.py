import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt
from ternarize import *
import sys, os
sys.path.append('/home/hyunjoon/github/additions_mxnet/rcnn')
from rcnn.symbol.proposal_target import *
from scipy.stats import norm

def inspect_weight_dist(prefix_net, epoch):
    #
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix_net, epoch)

    quantize_bit = 5

    err_log = {}
    err_uni = {}

    err_diff = []

    for k in sorted(arg_params):
        if not k.endswith('_weight'):
            continue
        v = arg_params[k].asnumpy().ravel()

        err_log[k] = measure_log_quantize_error(v, quantize_bit)
        err_uni[k] = measure_uni_quantize_error(v, quantize_bit)

        err_diff.append(err_log[k] - err_uni[k])

    plt.plot(range(len(err_diff)), err_diff)

    import ipdb
    ipdb.set_trace()

def measure_log_quantize_error(weights, quantize_bit):
    #
    th_big = np.std(weights) * 3.0
    iidx = np.where(np.abs(weights) < th_big)[0]
    weights = weights[iidx]
    
    quantize_exps = np.arange(2**(quantize_bit - 1) - 1)
    quantize_vals = 2**(quantize_exps + 0)
    quantize_vals = np.hstack((-quantize_vals[::-1], np.array((0)), quantize_vals)).astype(float)
    assert len(quantize_vals) == 2**quantize_bit - 1
    scaler = np.max(quantize_vals) / np.max(np.abs(weights))
    quantize_vals /= scaler
    return np.mean(comp_diff_weights(weights, quantize_vals))

def measure_uni_quantize_error(weights, quantize_bit):
    #
    th_big = np.std(weights) * 3.0
    iidx = np.where(np.abs(weights) < th_big)[0]
    weights = weights[iidx]
    
    quantize_vals = np.arange(2**(quantize_bit - 1) - 1) + 1
    quantize_vals = np.hstack((-quantize_vals[::-1], np.array((0)), quantize_vals)).astype(float)
    assert len(quantize_vals) == 2**quantize_bit - 1
    scaler = np.max(quantize_vals) / np.max(np.abs(weights))
    quantize_vals /= scaler
    return np.mean(comp_diff_weights(weights, quantize_vals))

def comp_diff_weights(weights, quantize_vals):
    diff_weights = np.full_like(weights, np.inf)
    for q in quantize_vals:
        d2 = (weights - q)**2.0
        midx = np.where(d2 < diff_weights)[0]
        diff_weights[midx] = d2[midx]
    return diff_weights

if __name__ == '__main__':
    #
    prefix_net = '/home/hyunjoon/github/additions_mxnet/rcnn/model/pvanet_voc07'
    epoch = 0

    inspect_weight_dist(prefix_net, epoch)
