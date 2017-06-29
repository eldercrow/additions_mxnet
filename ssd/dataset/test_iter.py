import mxnet as mx
import numpy as np
import cv2

class TestIter(mx.io.DataIter):
    """
    Detection Iterator, which will feed data and label to network
    Optional data augmentation is performed when providing batch

    Parameters:
    ----------
    imdb : Imdb
        image database
    data_shape : int or (int, int)
        image shape to be resized
    mean_pixels : float or float list
        [R, G, B], mean pixel values
    """
    def __init__(self, imdb, max_data_shapes, mean_pixels=[128, 128, 128], img_stride=32):
        super(TestIter, self).__init__()

        self._imdb = imdb
        self.batch_size = 1 # always 1
        self._max_data_shapes = max_data_shapes
        self._mean_pixels = mx.nd.array(mean_pixels).reshape((3,1,1))
        self._img_stride = img_stride

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
        return []

    def reset(self):
        self._current = 0

    def iter_next(self):
        return self._current < self._size

    def next(self):
        if self.iter_next():
            self._get_batch()
            data_batch = mx.io.DataBatch(data=self._data.values(),
                                   label=self._label.values(),
                                   provide_data=self.provide_data, provide_label=self.provide_label,
                                   pad=self.getpad(), index=self.getindex())
            self._current += self.batch_size
            return data_batch, self._im_info
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
        # batch_data = mx.nd.zeros((self.batch_size, 3, self._data_shape[0], self._data_shape[1]))
        assert self.batch_size == 1, "Batch size should be 1."

        # meaningless loop, kept just to preserve the original code
        for i in range(self.batch_size):
            if (self._current + i) >= self._size:
                if not self.is_train:
                    continue
                # use padding from middle in each epoch
                idx = (self._current + i + self._size // 2) % self._size
                index = self._index[idx]
            else:
                index = self._index[self._current + i]
            # index = self.debug_index
            im_path = self._imdb.image_path_from_index(index)
            with open(im_path, 'rb') as fp:
                img_content = fp.read()
            img = mx.img.imdecode(img_content)
            data, scale = self._data_augmentation(img)
            im_shape = img.shape
        batch_data = mx.nd.expand_dims(data, axis=0)
        scale = mx.nd.expand_dims(scale, axis=0)
        self._data = {'data': batch_data}
        self._label = {'label': None}
        self._im_info = {'im_scale': scale, 'im_path': im_path, 'im_shape': im_shape}

    def _data_augmentation(self, data):
        """
        perform data augmentations: resize, sub mean, swap channels
        """
        # first resize image w.r.t max image size
        sf_y = float(self._max_data_shapes[0]) / data.shape[0]
        sf_x = float(self._max_data_shapes[1]) / data.shape[1] 
        sf = np.minimum(1.0, np.minimum(sf_x, sf_y))
        sy = int(np.round(data.shape[0] * sf))
        sx = int(np.round(data.shape[1] * sf))
        sf_y = data.shape[0] / float(sy)
        sf_x = data.shape[1] / float(sx)
        if sy != data.shape[0] or sx != data.shape[1]:
            data = mx.img.imresize(data, sx, sy).asnumpy()
        else:
            data = data.asnumpy()
        # pad image w.r.t. image stride
        sy = np.ceil(sy / float(self._img_stride)) * self._img_stride
        sx = np.ceil(sx / float(self._img_stride)) * self._img_stride
        sy = int(np.maximum(sy, 384))
        sx = int(np.maximum(sx, 384))
        padded = np.reshape(self._mean_pixels.asnumpy(), (1, 1, 3))
        padded = np.tile(padded, (sy, sx, 1))
        padded[:(data.shape[0]), :(data.shape[1]), :] = data
        # data = mx.img.imresize(data, int(sx), int(sy)).asnumpy() # ignore slight aspect ratio break
        data = np.transpose(padded, (2, 0, 1)).astype(float)
        data = data - self._mean_pixels.asnumpy()
        return mx.nd.array(data), mx.nd.array((sf_y, sf_x))
