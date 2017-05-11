from __future__ import print_function
import os
from timeit import default_timer as timer
from dataset.testdb import TestDB
from dataset.face_test_iter import FaceTestIter
from mutable_module import MutableModule
import mxnet as mx
import numpy as np

class FaceDetector(object):
    """
    SSD detector which hold a detection network and wraps detection API

    Parameters:
    ----------
    symbol : mx.Symbol
        detection network Symbol
    model_prefix : str
        name prefix of trained model
    epoch : int
        load epoch of trained model
    data_shape : int
        input data resize shape
    mean_pixels : tuple of float
        (mean_r, mean_g, mean_b)
    batch_size : int
        run detection with batch size
    ctx : mx.ctx
        device to use, if None, use mx.cpu() as default context
    """

    def __init__(self,
                 symbol,
                 model_prefix,
                 epoch,
                 max_data_shapes,
                 mean_pixels,
                 img_stride=32,
                 th_nms=0.3333,
                 ctx=None):
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = mx.cpu()
        _, self.args, self.auxs = mx.model.load_checkpoint(model_prefix, epoch)
        assert max_data_shapes[0] % img_stride == 0 and max_data_shapes[1] % img_stride == 0
        self.max_data_shapes = max_data_shapes
        max_data_shapes = {
            'data': (1, 3, max_data_shapes[0], max_data_shapes[1])
        }
        # self.mod = mx.mod.Module(
        #     symbol,
        #     data_names=('data', 'im_scale'),
        #     label_names=None,
        #     context=ctx)
        self.mod = MutableModule(
            symbol,
            data_names = ('data', ),
            label_names=None,
            context=ctx,
            max_data_shapes=max_data_shapes)
        # self.data_shape = provide_data
        # self.mod.bind(data_shapes=max_data_shapes)
        # self.mod.set_params(args, auxs)
        self.mean_pixels = mean_pixels
        self.img_stride = img_stride
        self.th_nms = th_nms

    def detect(self, det_iter, show_timer=False):
        """
        detect all images in iterator

        Parameters:
        ----------
        det_iter : DetIter
            iterator for all testing images
        show_timer : Boolean
            whether to print out detection exec time

        Returns:
        ----------
        list of detection results
        """
        num_images = det_iter._size
        # if not isinstance(det_iter, mx.io.PrefetchingIter):
        #     det_iter = mx.io.PrefetchingIter(det_iter)
        if not self.mod.binded:
            self.mod.bind(
                data_shapes=[det_iter.provide_data[0]],
                label_shapes=None,
                for_training=False,
                force_rebind=True)
            self.mod.set_params(self.args, self.auxs)
        start = timer()
        result = []
        im_paths = []
        detections = []
        for i, (datum, im_info) in enumerate(det_iter):
            self.mod.forward(datum)
            out = self.mod.get_outputs()
            # time_elapsed = timer() - start
            im_scale = im_info['im_scale'][0].asnumpy()
            n_dets = int(out[1].asnumpy()[0])
            im_paths.append(im_info['im_path'])
            if n_dets == 0:
                result.append(np.zeros((0, 6)))
                continue
            dets = out[0][0][:n_dets]
            # dets = self._transform_roi(dets)
            # vidx = self._do_nms(dets)
            vdets = dets.asnumpy()
            # vidx = vdets[:, 1] > 0
            # vdets = vdets[vidx, :]
            vdets[:, 2] *= im_scale[1]
            vdets[:, 3] *= im_scale[0]
            vdets[:, 4] *= im_scale[1]
            vdets[:, 5] *= im_scale[0]
            result.append(vdets)

            if i % 10 == 0:
                print('Processing image {}/{}, {} faces detected.'.format(i+1, num_images, n_dets))
        #     detections.append(out[0][0][:n_dets].asnumpy())
        # for i in range(len(detections)):
        #     dets = mx.nd.array(detections[i])
        #     dets = self._transform_roi(dets)
        #     vidx = self._do_nms(dets)
        #     vdets = dets.asnumpy()
        #     vdets = vdets[vidx, :]
        #     result.append(vdets)
        time_elapsed = timer() - start
        if show_timer:
            print("Detection time for {} images: {:.4f} sec".format(
                num_images, time_elapsed))
        return result, im_paths

    def im_detect(self,
                  im_list,
                  root_dir=None,
                  extension=None,
                  show_timer=False):
        """
        wrapper for detecting multiple images

        Parameters:
        ----------
        im_list : list of str
            image path or list of image paths
        root_dir : str
            directory of input images, optional if image path already
            has full directory information
        extension : str
            image extension, eg. ".jpg", optional

        Returns:
        ----------
        list of detection results in format [det0, det1...], det is in
        format np.array([id, score, xmin, ymin, xmax, ymax]...)
        """
        test_db = TestDB(im_list, root_dir=root_dir, extension=extension)
        test_iter = FaceTestIter(test_db, self.max_data_shapes,
                                 self.mean_pixels, self.img_stride)
        return self.detect(test_iter, show_timer)

    def visualize_detection(self, img, dets, classes=[], thresh=0.6):
        """
        visualize detections in one image

        Parameters:
        ----------
        img : numpy.array
            image, in bgr format
        dets : numpy.array
            ssd detections, numpy.array([[id, score, x1, y1, x2, y2]...])
            each row is one object
        classes : tuple or list of str
            class names
        thresh : float
            score threshold
        """
        import matplotlib.pyplot as plt
        import random
        plt.imshow(img)
        # height = img.shape[0]
        # width = img.shape[1]
        # wr = width / float(self.data_shape[1])
        # hr = height / float(self.data_shape[0])
        colors = dict()
        for i in range(dets.shape[0]):
            cls_id = int(dets[i, 0])
            if cls_id >= 0:
                score = dets[i, 1]
                if score > thresh:
                    if cls_id not in colors:
                        colors[cls_id] = (random.random(), random.random(),
                                          random.random())
                    # xmin = int(dets[i, 2] * width)
                    # ymin = int(dets[i, 3] * height)
                    # xmax = int(dets[i, 4] * width)
                    # ymax = int(dets[i, 5] * height)
                    xmin = int(dets[i, 2])
                    ymin = int(dets[i, 3])
                    xmax = int(dets[i, 4])
                    ymax = int(dets[i, 5])
                    rect = plt.Rectangle(
                        (xmin, ymin),
                        xmax - xmin,
                        ymax - ymin,
                        fill=False,
                        edgecolor=colors[cls_id],
                        linewidth=2.5)
                    plt.gca().add_patch(rect)
                    class_name = str(cls_id)
                    if classes and len(classes) > cls_id:
                        class_name = classes[cls_id]
                    # plt.gca().text(
                    #     xmin,
                    #     ymin - 2,
                    #     '{:.3f}'.format(score),
                    #     bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                    #     fontsize=7,
                    #     color='white')
                    # plt.gca().text(
                    #     xmin,
                    #     ymin - 2,
                    #     '{:s} {:.3f}'.format(class_name, score),
                    #     bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                    #     fontsize=7,
                    #     color='white')
        plt.show()

    def detect_and_visualize(self,
                             im_list,
                             root_dir=None,
                             extension=None,
                             classes=[],
                             thresh=0.6,
                             show_timer=False):
        """
        wrapper for im_detect and visualize_detection

        Parameters:
        ----------
        im_list : list of str or str
            image path or list of image paths
        root_dir : str or None
            directory of input images, optional if image path already
            has full directory information
        extension : str or None
            image extension, eg. ".jpg", optional

        Returns:
        ----------

        """
        import cv2
        dets, _ = self.im_detect(
            im_list, root_dir, extension, show_timer=show_timer)
        root_dir = '' if not root_dir else root_dir
        extension = '' if not extension else extension
        if not isinstance(im_list, list):
            im_list = [im_list]
        assert len(dets) == len(im_list)
        for k, det in enumerate(dets):
            fn_img = os.path.join(root_dir, im_list[k] + extension)
            img = cv2.imread(fn_img)
            img[:, :, (0, 1, 2)] = img[:, :, (2, 1, 0)]
            self.visualize_detection(img, det, classes, thresh)

    def _transform_roi(self, dets, ratio=0.8):
        #
        dets_t = mx.nd.transpose(dets, axes=(1,0))
        cx = (dets_t[6] + dets_t[8]) * 0.5
        cy = (dets_t[7] + dets_t[9]) * 0.5
        aw = (dets_t[8] - dets_t[6])
        aw *= ratio
        ah = (dets_t[9] - dets_t[7])
        cx += dets_t[2] * aw
        cy += dets_t[3] * ah
        w = (2.0**dets_t[4]) * aw
        h = (2.0**dets_t[5]) * ah
        dets_t[2] = cx - w / 2.0
        dets_t[3] = cy - h / 2.0
        dets_t[4] = cx + w / 2.0
        dets_t[5] = cy + h / 2.0
        return mx.nd.transpose(dets_t[:6], axes=(1, 0))

    def _do_nms(self, dets):
        #
        dets_t = mx.nd.transpose(dets, axes=(1,0))
        areas_t = (dets_t[4] - dets_t[2]) * (dets_t[5] - dets_t[3])

        vmask = np.ones((dets.shape[0],), dtype=int)
        vidx = []
        
        for i in range(dets.shape[0]):
            if vmask[i] == 0:
                continue
            iw = mx.nd.minimum(dets[i][4], dets_t[4]) - mx.nd.maximum(dets[i][2], dets_t[2])
            ih = mx.nd.minimum(dets[i][5], dets_t[5]) - mx.nd.maximum(dets[i][3], dets_t[3])
            I = mx.nd.maximum(iw, 0) * mx.nd.maximum(ih, 0)
            iou = (I / mx.nd.maximum(areas_t + areas_t[i] - I, 1e-06)).asnumpy()
            nidx = np.where(iou > self.th_nms)[0] 
            vmask[nidx] = 0
            vidx.append(i)
        return vidx
