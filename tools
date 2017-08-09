import mxnet as mx
import numpy as np

def crop_roi_patch(img, roi):
    hh = img.shape[0]
    ww = img.shape[1]
    if roi[0] >= 0 and roi[1] >= 0 and roi[2] <= ww and roi[3] <= hh:
        patch = mx.img.fixed_crop(mx.nd.array(img), roi[0], roi[1], roi[2]-roi[0], roi[3]-roi[1])
    else:
        # padding
        pw = roi[2] - roi[0]
        ph = roi[3] - roi[1]
        # roi in image
        li = np.maximum(0, roi[0])
        ri = np.minimum(ww, roi[2])
        ui = np.maximum(0, roi[1])
        bi = np.minimum(hh, roi[3])
        # roi in patch
        lp = np.maximum(0, -roi[0])
        up_ = np.maximum(0, -roi[1])
        rp = lp + (ri - li)
        bp = up_ + (bi - ui)

        patch_ = np.full((ph, pw, 3), 128, dtype=np.uint8)
        patch_[up_:bp, lp:rp, :] = img[ui:bi, li:ri, :]
        patch = mx.nd.array(patch_)
    # patch = mx.nd.transpose(patch, axes=(2, 0, 1))
    # patch = patch.astype('float32')
    # patch = patch - self._mean_pixels
    return patch

