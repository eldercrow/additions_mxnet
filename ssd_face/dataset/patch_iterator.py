import mxnet as mx
import numpy as np
import cv2
from tools.rand_sampler import RandSampler
import matplotlib.pyplot as plt

class PatchIter(mx.io.DataIter):
    """
    Patch generation iterator, which will feed patches and their labels.
    1. Load an image.
    2. For each face, crop a patch centered by the face (230x230 for pvanet and hjnet).
    3. Positive roi should be at the center, ignore other faces, draw negative rois within the patch.
    4. We can even compute matching anchors for positive/negative rois.
        * anchor locations: (scale, x, y)
    5. Compute class labels, and regression targets for positive anchors.
    6. Iterate 1-6 to sample patches until we have one batch, then provide the batch.

    Parameters:
    ----------
    imdb : Imdb
        image database
    batch_size : int
        batch size
    patch_shape : int or (int, int)
        patch shape 
    mean_pixels : float or float list
        [R, G, B], mean pixel values
    rand_samplers : list
        random cropping sampler list, if not specified, will
        use original image only
    rand_mirror : bool
        whether to randomly mirror input images, default False
    shuffle : bool
        whether to shuffle initial image list, default False
    rand_seed : int or None
        whether to use fixed random seed, default None
    max_crop_trial : bool
        if random crop is enabled, defines the maximum trial time
        if trial exceed this number, will give up cropping
    is_train : bool
        whether in training phase, default True, if False, labels might
        be ignored
    """
    def __init__(self, imdb, batch_size, patch_shape, \
                 mean_pixels=[128, 128, 128], rand_samplers=[], \
                 rand_mirror=False, shuffle=False, rand_seed=None, \
                 is_train=True, max_crop_trial=50):
        super(PatchIter, self).__init__()

        self._imdb = imdb
        self.batch_size = batch_size
        if isinstance(patch_shape, int):
            patch_shape = (patch_shape, patch_shape)
        self._data_shape = patch_shape
        self._mean_pixels = mx.nd.array(mean_pixels).reshape((3,1,1))
        if not rand_samplers:
            self._rand_samplers = []
        else:
            if not isinstance(rand_samplers, list):
                rand_samplers = [rand_samplers]
            assert isinstance(rand_samplers[0], RandSampler), "Invalid rand sampler"
            self._rand_samplers = rand_samplers
        self.is_train = is_train
        self._rand_mirror = rand_mirror
        self._shuffle = shuffle
        if rand_seed:
            np.random.seed(rand_seed) # fix random seed
        self._max_crop_trial = max_crop_trial

        self._current = 0
        self._size = imdb.num_images
        self._index = np.arange(self._size)

        self._data = None
        self._label = None
        self._get_batch()

    @property
    def provide_data(self):
        return [(k, v.shape) for k, v in self._data.items()]

    @property
    def provide_label(self):
        if self.is_train:
            return [(k, v.shape) for k, v in self._label.items()]
        else:
            return []

    def reset(self):
        self._current = 0
        # if self._shuffle:
        #     np.random.shuffle(self._index)
        self._imdb.reset_patch()
        self._size = self._imdb.num_images
        self._index = np.arange(self._size)

    def iter_next(self):
        return self._current < self._size

    def next(self):
        if self.iter_next():
            self._get_batch()
            data_batch = mx.io.DataBatch(data=self._data.values(),
                                   label=self._label.values(),
                                   pad=self.getpad(), index=self.getindex())
            self._current += self.batch_size
            return data_batch
        else:
            raise StopIteration

    def getindex(self):
        return self._current // self.batch_size

    def getpad(self):
        pad = self._current + self.batch_size - self._size
        return 0 if pad < 0 else pad
    
    def _get_batch(self):
        """
        Load data/label from dataset
        """
        batch_data = mx.nd.zeros((self.batch_size, 3, self._data_shape[0], self._data_shape[1]))
        batch_path = []
        batch_label = []
        # first gather image paths and labels for all patche in a batch
        for i in range(self.batch_size):
            if (self._current + i) >= self._size:
                # use padding from middle in each epoch
                idx = (self._current + i + self._size // 2) % self._size
                index = self._index[idx]
            else:
                index = self._index[self._current + i]
            # index = self.debug_index
            im_path = self._imdb.image_path_from_index(index)
            batch_path.append(im_path)
            gt = self._imdb.label_from_index(index)
            batch_label.append(gt)
        # Load images that will be used in this batch, with some sanity check
        batch_label = np.array(batch_label)
        curr_path = ''
        curr_scale = 0.0
        batch_imgs = []
        patch2img = []
        curr_idx = -1
        for p, l in zip(batch_path, batch_label):
            if p != curr_path:
                curr_path = p
                curr_scale = l[0]
                with open(p, 'rb') as fh:
                    img_content = fh.read()
                img = mx.img.imdecode(img_content)
                if l[0] != 1:
                    hh = int(np.round(img.shape[0] * l[0]))
                    ww = int(np.round(img.shape[1] * l[0]))
                    img = mx.img.imresize(img, ww, hh, interp=2)
                batch_imgs.append(img.asnumpy())
                curr_idx += 1
                patch2img.append(curr_idx)
            else:
                assert l[0] == curr_scale, 'patches are not well arranged'
                patch2img.append(curr_idx)
        # crop patches from images 
        for i, (pidx, l) in enumerate(zip(patch2img, batch_label)):
            patch = self._crop_patch(batch_imgs[pidx], l[6:].astype(int))
            batch_data[i] = patch

        self._data = {'data': batch_data}
        if self.is_train:
            self._label = {'label': mx.nd.array(np.array(batch_label[:, 1:6]))}
        else:
            self._label = {'label': None}

    def _crop_patch(self, img, roi):
        hh = img.shape[0]
        ww = img.shape[1]
        if roi[0] >= 0 and roi[1] >= 0 and roi[2] <= ww and roi[3] <= hh:
            patch = mx.img.fixed_crop(mx.nd.array(img), roi[0], roi[1], roi[2]-roi[0], roi[3]-roi[1])
        else:
            # padding
            pw = roi[2] - roi[0]
            ph = roi[3] - roi[1]
            assert pw == self._data_shape[0] and ph == self._data_shape[1]
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
        # import ipdb
        # plt.imshow(patch.asnumpy())
        # print label
        # ipdb.set_trace()
        patch = mx.nd.transpose(patch, axes=(2, 0, 1))
        patch = patch.astype('float32')
        patch = patch - self._mean_pixels
        return patch
