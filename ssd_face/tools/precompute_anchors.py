import mxnet as mx

def precompute_anchors(sizes, ratios, init_shape, n_octave):
    ''' precompute all anchors at once
    '''
    h, w = init_shape
    n_anchors = len(sizes) * len(ratios)

    anchors_all = np.empty((0, 4))

    for ii in range(n_octave):
        # compute center positions
        x = np.linspace(0.0, 1.0, w+1)
        y = np.linspace(0.0, 1.0, h+1)
        xv, yv = np.meshgrid(x, y)
        xv = xv[0:-1, 0:-1] + 0.5 / w
        yv = yv[0:-1, 0:-1] + 0.5 / h

        # compute heights and widths
        wh = np.zeros((n_anchors, 2))
        k = 0
        for s in sizes:
            for r in ratios:
                r2 = np.sqrt(float(r))
                wh[k, 0] = s / r2
                wh[k, 1] = s * r2
                k += 1

        # build anchors
        anchors = np.zeros((h, w, n_anchors*4))
        for i in range(n_anchor):
            anchors[:, :, 4*i+0] = xv - wh[i, 0] / 2.0
            anchors[:, :, 4*i+1] = yv - wh[i, 0] / 2.0
            anchors[:, :, 4*i+2] = xv + wh[i, 0] / 2.0
            anchors[:, :, 4*i+3] = yv + wh[i, 1] / 2.0

        anchors_all = np.vstack((anchors_all, anchors))
        h /= 2
        w /= 2
        sizes = [s * 2.0 for s in sizes]

    return anchors_all
