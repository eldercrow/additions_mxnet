import numpy as np

def do_nms(dets_all, n_class, th_nms):
    '''
    dets: detection results, (n_dets, 6)
    each row of dets: (class_id, class_prob, xmin, ymin, xmax, ymax)
    '''
    cidx_all = []
    for c in range(n_class):
        cidx = np.where(dets_all[:, 0] == c)[0]
        if cidx.size == 0:
            continue
        dets = dets_all[cidx, :]
        sidx = np.argsort(dets[:, 1])[::-1]
        cidx = cidx[sidx]
        dets = dets[sidx, :]
        vidx = _nms(dets, th_nms)
        cidx = cidx[vidx]
        cidx_all.append(cidx)
    try:
        if cidx_all:
            cidx_all = np.hstack((cidx_all))
        nidx = np.setdiff1d(np.arange(dets_all.shape[0]), cidx_all)
    except:
        import ipdb
        ipdb.set_trace()
    dets_all[nidx, 0] = -1
    return dets_all


def _nms(dets, th_nms):
    '''
    '''
    areas = (dets[:, 4] - dets[:, 2]) * (dets[:, 5] - dets[:, 3])
    vmask = np.ones((dets.shape[0],), dtype=int)
    vidx = []
    for i, d in enumerate(dets):
        if vmask[i] == 0:
            continue
        iw = np.minimum(d[4], dets[i:, 4]) - np.maximum(d[2], dets[i:, 2])
        ih = np.minimum(d[5], dets[i:, 5]) - np.maximum(d[3], dets[i:, 3])
        I = np.maximum(iw, 0) * np.maximum(ih, 0)
        iou = I / np.maximum(areas[i:] + areas[i] - I, 1e-08)
        nidx = np.where(iou > th_nms)[0] + i
        vmask[nidx] = 0
        vidx.append(i)
    return vidx
