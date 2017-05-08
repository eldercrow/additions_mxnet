from __future__ import print_function
import os
import mxnet as mx
import numpy as np
from imdb import Imdb
import xml.etree.ElementTree as ET
# from evaluate.eval_voc import voc_eval
import cv2
import cPickle
from tools.crop_roi_patch import crop_roi_patch

from multiprocessing import Process, Queue


class WiderPatch(Imdb):
    """
    Create fixed size patches from the WIDER face dataset.
    Initially, all the image paths and labels are loded.
    Then whenever needed, draw fixed size patches from all the images, adjust labels accordingly, 
    and feed them to the iterator.

    Parameters:
    ----------
    image_set : str
        set to be used, can be train, val, trainval, test
    devkit_path : str
        devkit path of VOC dataset
    shuffle : boolean
        whether to initial shuffle the image list
    is_train : boolean
        if true, will load annotations
    patch_shape : int or (int, int)
    min_roi_size : int 
        objects whose shorter sizes are smaller than it will not be sampled
    max_roi_size : int 
        objects whose longer sizes are bigger than it will not be sampled
    max_patch_per_image : int
        maximum number of patches to be sampled from each image.
    range_rand_scale : (float, float)
        minimum and maximum scale ratios. (0, 1] for min, [1, 3] for max, 
        None for no random scaling.
    max_crop_trial : bool
        if random crop is enabled, defines the maximum trial time
        if trial exceed this number, will give up cropping
    """
    IDX_VER = '170422_2' # for caching

    def __init__(self, image_set, devkit_path, shuffle=True, is_train=True, **kwargs):
        super(WiderPatch, self).__init__('widerpatch_' + image_set) # e.g. wider_trainval
        self.image_set = image_set
        self.devkit_path = devkit_path
        self.data_path = devkit_path # os.path.join(devkit_path, 'wider')
        self.extension = '.jpg'
        assert is_train == True, 'Only for training!'

        self.classes = ['__background__', 'face',]

        self.config = { \
                'patch_shape': 256, 
                'min_roi_size': 8, 
                'max_roi_size': 256,
                'range_rand_scale': None,
                'max_crop_trial': 50,
                'max_patch_per_image': 16, 
                'use_difficult': False
                }
        for k, v in kwargs.iteritems():
            assert k in self.config, 'Unknown parameter %s.' % k
            self.config[k] = v

        self.patch_shape = self.config['patch_shape']
        self.min_roi_size = self.config['min_roi_size']
        self.max_roi_size = self.config['max_roi_size']
        self.range_rand_scale = self.config['range_rand_scale']
        self.max_crop_trial = self.config['max_crop_trial']
        self.max_patch_per_image = self.config['max_patch_per_image']

        self.num_classes = len(self.classes) 
        self.max_objects = 0

        # try to load cached data
        cached = self._load_from_cache()
        if cached is None: # no cached data, load from DB (and save)
            fn_cache = os.path.join(self.cache_path, self.name + '_' + self.IDX_VER + '.pkl')
            self.image_set_index = self._load_image_set_index(shuffle)
            self.num_orig_images = len(self.image_set_index)
            self.labels, self.img_shapes = self._load_image_labels()
            self._save_to_cache()
        else:
            self.image_set_index = cached['image_set_index']
            self.num_orig_images = len(self.image_set_index)
            if 'labels' in cached and 'img_shapes' in cached:
                self.labels = cached['labels']
                self.img_shapes = cached['img_shapes']
            else:
                self.labels, self.img_shapes = self._load_image_labels()
                self._save_to_cache()

        self.patch_im_path = None
        self.patch_im_scale = None
        self.patch_labels = None
        self.patch_im_path, self.patch_labels = self._build_patch_db()
        self.num_images = len(self.patch_im_path)
        # self._debug_save_patches()
        # prepare for the next epoch
        self.data_queue = Queue()
        # self.p = Process(target=self._build_next_patch_db, args=(self.data_queue,))
        # self.p.start()

    @property
    def cache_path(self):
        """
        make a directory to store all caches

        Returns:
        ---------
            cache path
        """
        cache_path = os.path.join(os.path.dirname(__file__), '..', 'cache')
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)
        return cache_path

    def reset_patch(self):
        # is data for the next epoch ready?
        if self.data_queue.empty(): 
            return
        # self.patch_im_path, self.patch_labels = self.data_queue.get()
        # self.num_images = len(self.patch_im_path)
        # if self.p.is_alive():
        #     self.p.terminate()
        # self.p = Process(target=self._build_next_patch_db, args=(self.data_queue,))
        # self.p.start()

    def _load_from_cache(self):
        fn_cache = os.path.join(self.cache_path, self.name + '_' + self.IDX_VER + '.pkl')
        cached = {}
        if os.path.exists(fn_cache):
            try:
                with open(fn_cache, 'rb') as fh:
                    header = cPickle.load(fh)
                    assert header['ver'] == self.IDX_VER, "Version mismatch, re-index DB."
                    self.max_objects = header['max_objects']
                    iidx = cPickle.load(fh)
                    cached['image_set_index'] = iidx['image_set_index']
                    img_shapes = cPickle.load(fh)
                    cached['img_shapes'] = img_shapes['img_shapes']
                    labels = cPickle.load(fh)
                    cached['labels'] = labels['labels']
            except: 
                # print 'Exception in load_from_cache.'
                return None

        return None if not cached else cached

    def _save_to_cache(self):
        fn_cache = os.path.join(self.cache_path, self.name + '_' + self.IDX_VER + '.pkl')
        with open(fn_cache, 'wb') as fh:
            cPickle.dump({'ver': self.IDX_VER, 'max_objects': self.max_objects}, fh, cPickle.HIGHEST_PROTOCOL)
            cPickle.dump({'image_set_index': self.image_set_index}, fh, cPickle.HIGHEST_PROTOCOL)
            cPickle.dump({'img_shapes': self.img_shapes}, fh, cPickle.HIGHEST_PROTOCOL)
            cPickle.dump({'labels': self.labels}, fh, cPickle.HIGHEST_PROTOCOL)

    def _load_image_set_index(self, shuffle):
        """
        find out which indexes correspond to given image set (train or val)

        Parameters:
        ----------
        shuffle : boolean
            whether to shuffle the image list
        Returns:
        ----------
        entire list of images specified in the setting
        """
        image_set_index_file = os.path.join(self.data_path, 'img', self.image_set + '.txt')
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        with open(image_set_index_file) as f:
            image_set_index = [x.strip() for x in f.readlines()]
        if shuffle:
            np.random.shuffle(image_set_index)
        return image_set_index

    # we will index image path and label using other functions, 
    # the original functions will be used to refer patch based info.
    def full_image_path_from_index(self, index):
        """
        given image index, find out full path

        Parameters:
        ----------
        index: int
            index of a specific image
        Returns:
        ----------
        full path of this image
        """
        assert self.image_set_index is not None, "Dataset not initialized"
        name = self.image_set_index[index]
        image_file = os.path.join(self.data_path, 'img', name + self.extension)
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file

    def full_label_from_index(self, index):
        assert self.labels is not None, "Imagewise labels not processed"
        return self.labels[index]

    def image_path_from_index(self, index):
        '''
        given patch index, find out full path of the image
        containng the patche
        '''
        assert self.patch_im_path is not None, "Dataset not initialized"
        name = self.patch_im_path[index]
        image_file = os.path.join(self.data_path, 'img', name + self.extension)
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file

    def img_shape_from_index(self, index):
        assert self.img_shapes is not None, "Dataset not initialized"
        return self.img_shapes[index, 0], self.img_shapes[index, 1] # (ww_img, hh_img)

    def label_from_index(self, index):
        """
        given patch index, return preprocessed ground-truth

        Parameters:
        ----------
        index: int
            index of a specific image
        Returns:
        ----------
        ground-truths of this image
        """
        assert self.patch_labels is not None, "Patchwise labels not processed"
        return self.patch_labels[index, :].tolist()

    def _label_path_from_index(self, index):
        """
        given image index, find out annotation path

        Parameters:
        ----------
        index: int
            index of a specific image

        Returns:
        ----------
        full path of annotation file
        """
        name = self.image_set_index[index]
        bb_file = os.path.join(self.data_path, 'annotation', name + '.bb')
        prop_file = os.path.join(self.data_path, 'annotation', name + '.prop_label')
        assert os.path.exists(bb_file), 'Path does not exist: {}'.format(bb_file)
        return bb_file, prop_file

    def _load_image_labels(self):
        """
        preprocess all ground-truths

        Returns:
        ----------
        labels packed in [num_images x max_num_objects x 5] tensor
        """
        labels = []
        img_shapes = []
        max_objects = 0
        max_label = np.empty((0, 5))
        max_fn = ''

        # load ground-truth from xml annotations
        for idx in range(len(self.image_set_index)):
            bb_file, prop_file = self._label_path_from_index(idx)
            cls_id = self.classes.index('face')
            bbs = np.reshape(np.loadtxt(bb_file).astype(float), (-1, 4))
            if bbs.size == 0: 
                labels.append(np.empty((0, 5)))
                continue
            ww_img = 0
            hh_img = 0
            small_mask = np.minimum(bbs[:, 2], bbs[:, 3]) < self.config['min_roi_size']
            # remove bbs that are 1) invalid or 2) too small and occluded.
            with open(prop_file, 'r') as fh:
                prop_data = fh.read().splitlines()
            for pdata in prop_data:
                prop_bb = pdata.split(' ')
                if prop_bb[0] == 'invalid_label_list':
                    invalid_mask = np.array(prop_bb[1:]).astype(int) == 1
                if prop_bb[0] == 'occlusion_label_list':
                    small_mask = np.logical_and(small_mask, np.array(prop_bb[1:]).astype(int) == 2)
                # also get image size
                if prop_bb[0] == 'image_size':
                    ww_img = int(prop_bb[1])
                    hh_img = int(prop_bb[2])
            if self.config['use_difficult'] is not True:
                invalid_mask = np.logical_or(invalid_mask, small_mask)
            # FOR DEBUG, save image size to .prop_label
            if ww_img == 0:
                fn_img = self.full_image_path_from_index(idx)
                img = cv2.imread(fn_img)
                prop_data.append('image_size %d %d' % (img.shape[1], img.shape[0]))
                with open(prop_file, 'w') as fh:
                    for pdata in prop_data:
                        fh.write(pdata + '\n')
                ww_img = img.shape[1]
                hh_img = img.shape[0]

            img_shapes.append((ww_img, hh_img))
            valid_idx = np.where(invalid_mask == False)[0]
            bbs = bbs[valid_idx, :]
            if bbs.size == 0: 
                labels.append(np.empty((0, 5)))
                continue
            # we need [xmin, ymin, xmax, ymax], but wider DB has [xmin, ymin, width, height]
            # bbs[:, 2] += bbs[:, 0] - 1.0
            # bbs[:, 3] += bbs[:, 1] - 1.0
            # # normalize to [0, 1]
            # bbs[:, 0::2] /= ww_img
            # bbs[:, 1::2] /= hh_img
            #
            # bbs = np.minimum(np.maximum(bbs, 0.0), 1.0)

            label = np.zeros((bbs.shape[0], 5))
            label[:, 0] = cls_id
            label[:, 1:] = bbs
            labels.append(label)
            if label.shape[0] > max_objects:
                max_objects = label.shape[0]

        assert max_objects > 0, "No objects found for any of the images"
        return labels, np.array(img_shapes, dtype=np.int32)

    def _build_patch_db(self):
        im_paths = []
        patch_labels = np.empty((0, 10))
        for i in range(self.num_orig_images):
            im_path, patch_label = self._sample_patches(i)
            if im_path is None:
                continue
            n_patch = patch_label.shape[0]
            im_paths += [im_path] * n_patch
            patch_labels = np.vstack((patch_labels, patch_label))
            # if i % 500 == 1:
            #     print('processing image {}'.format(i))
        return im_paths, patch_labels
        # self.patch_im_path = im_paths
        # self.patch_labels = patch_labels
        # self.num_images = self.patch_labels.shape[0]

    def _build_next_patch_db(self, data_queue):
        next_patch_im_path, next_patch_labels = self._build_patch_db()
        data_queue.put([next_patch_im_path, next_patch_labels])
        print('Data for the next epoch is ready.')

    def _sample_patches(self, index):
        """ 
        sample face patches from self.image_set_index[index] 

        Returns:
        -------
        im_path: relative path of the image, w/o extension
        scaler: random scale factor applied to this image
        labels: n * (pos/neg label, 
                     xmin_patch, ymin, xmax, ymax, 
                     xmin_target_roi, ymin, xmax, ymax)
        """
        im_path = self.image_set_index[index]
        ww_img, hh_img = self.img_shape_from_index(index)
        gt_label = self.full_label_from_index(index).copy()

        # skip invalid image
        if gt_label.size == 0:
            return None, None

        # apply random scaling
        if self.range_rand_scale is not None:
            l = self.range_rand_scale[0]
            h = self.range_rand_scale[1]
            scaler = np.random.uniform(low=l, high=h, size=(1,))[0]
            ww_img *= scaler
            hh_img *= scaler
            # new_ww = np.round(ww_img * scaler)
            # new_hh = np.round(hh_img * scaler)
            # img = mx.img.imresize(img, new_ww, new_hh)
            gt_label[:, 1:3] = (gt_label[:, 1:3] + 0.5) * scaler - 0.5
            gt_label[:, 3:] *= scaler
        else:
            scaler = 1

        # remove too big or too small bbs
        vidx = np.min(gt_label[:, 3:], axis=1) >= self.min_roi_size
        vidx = np.logical_and(vidx, np.max(gt_label[:, 3:], axis=1) <= self.max_roi_size)
        vidx = np.where(vidx)[0]
        gt_label = gt_label[vidx, :]
        if gt_label.size == 0:
            return None, None
        # random sample gt bbs if there are too many
        np.random.shuffle(gt_label)
        if gt_label.shape[0] > self.max_patch_per_image:
            gt_label_pos = gt_label[:self.max_patch_per_image, :]
        else:
            gt_label_pos = gt_label

        # sample patch roi for each gt bb
        pos_patch_rois, _ = _draw_random_trans_patches(gt_label_pos[:, 1:], 0.65, 1.0, self.patch_shape)
        pos_labels = np.zeros((gt_label_pos.shape[0], 9))
        pos_labels[:, 0:5] = gt_label_pos
        pos_labels[:, 5:] = pos_patch_rois

        # draw negative samples
        # 1. pure random sample patches
        n_neg_patch_rois = gt_label_pos.shape[0] * 3
        neg_patch_rois = np.zeros((n_neg_patch_rois, 4))
        # for now we use the format (xmin, ymin, width, height)
        neg_patch_rois[:, 0:2] = np.random.uniform(low=0.0, high=1.0, size=(n_neg_patch_rois, 2))
        neg_patch_rois[:, 0] = neg_patch_rois[:, 0] * ww_img
        neg_patch_rois[:, 1] = neg_patch_rois[:, 1] * hh_img
        neg_patch_rois[:, 2:4] = self.patch_shape

        # for pure random sample patches, randomly map a target
        tsize = np.random.uniform(low=self.min_roi_size, high=self.max_roi_size, 
                size=(neg_patch_rois.shape[0], 1))
        tratio = np.random.uniform(low=0.75, high=1.0, size=(neg_patch_rois.shape[0], 1))
        ww_neg = tsize
        hh_neg = tsize*tratio
        swap_mask = np.random.uniform(low=0, high=1, size=(neg_patch_rois.shape[0], 1)) > 0.5
        swap_mask = np.where(swap_mask == True)[0]
        cx_neg = neg_patch_rois[:, 0:1] + (self.patch_shape - 1.0) / 2.0
        cy_neg = neg_patch_rois[:, 1:2] + (self.patch_shape - 1.0) / 2.0
        neg_rois = np.hstack((cx_neg, cy_neg, ww_neg, hh_neg))
        neg_rois[swap_mask, 3:4] = ww_neg[swap_mask, :]
        neg_rois[swap_mask, 2:3] = hh_neg[swap_mask, :]
        neg_rois[:, 0] -= (neg_rois[:, 2] - 1.0) / 2.0
        neg_rois[:, 1] -= (neg_rois[:, 3] - 1.0) / 2.0

        # sanity check
        neg_patch_rois, neg_rois, _ = \
                self._check_negative_patches(neg_patch_rois, neg_rois, ww_img, hh_img, gt_label[:, 1:])

        neg_labels = np.zeros((neg_rois.shape[0], 9))
        neg_labels[:, 1:5] = neg_rois
        neg_labels[:, 5:] = neg_patch_rois
        if neg_labels.shape[0] > pos_labels.shape[0]:
            neg_labels = neg_labels[:pos_labels.shape[0], :]

        # 2. semi hard negatives
        hard_neg_patch_rois = np.empty((0, 4))
        hard_neg_rois = np.empty((0, 4))
        for i in range(3):
            nprois, nrois = _draw_random_trans_patches(gt_label_pos[:, 1:], -0.05, 0.3, self.patch_shape)
            nprois, nrois, _ = self._check_negative_patches(nprois, nrois, ww_img, hh_img, gt_label[:, 1:])
            hard_neg_patch_rois = np.vstack((hard_neg_patch_rois, nprois))
            hard_neg_rois = np.vstack((hard_neg_rois, nrois))

        hard_neg_labels = np.zeros((hard_neg_rois.shape[0], 9))
        hard_neg_labels[:, 1:5] = hard_neg_rois
        hard_neg_labels[:, 5:] = hard_neg_patch_rois
        if hard_neg_labels.shape[0] > pos_labels.shape[0] * 2:
            hard_neg_labels = hard_neg_labels[:pos_labels.shape[0] * 2, :]

        rois = np.random.permutation(np.vstack((pos_labels, neg_labels, hard_neg_labels)))
        # make rois ralative to patch rois
        rois[:, 1] -= rois[:, 5]
        rois[:, 2] -= rois[:, 6]
        # TODO: test, make roi square
        cx = rois[:, 1] + (rois[:, 3] - 1.0) / 2.0
        cy = rois[:, 2] + (rois[:, 4] - 1.0) / 2.0
        max_roi_sz = np.maximum(rois[:, 3], rois[:, 4])
        rois[:, 1] = cx - max_roi_sz / 2.0
        rois[:, 2] = cy - max_roi_sz / 2.0
        rois[:, 3] = max_roi_sz
        rois[:, 4] = max_roi_sz
        # convert to [xmin, ymin, xmax, ymax]
        rois[:, 3] += rois[:, 1]
        rois[:, 4] += rois[:, 2]
        rois[:, 7] += rois[:, 5]
        rois[:, 8] += rois[:, 6]
        rois = np.round(rois)
        rois[:, 1:5] /= self.patch_shape
        scaler = np.tile(np.reshape(np.array(scaler), (1,1)), (rois.shape[0], 1))
        rois = np.hstack((scaler, rois))
        return im_path, rois

    def _check_negative_patches(self, neg_patch_rois, neg_rois, ww_img, hh_img, gt_label):
        # OOB check
        img_roi = np.array([0, 0, ww_img, hh_img])
        overlap_img = _compute_overlap(np.reshape(img_roi, (1, 4)), neg_patch_rois).ravel()
        iidx1 = np.where(overlap_img > 0.5)[0]
        neg_patch_rois = neg_patch_rois[iidx1, :]
        neg_rois = neg_rois[iidx1, :]

        # iou check
        iou_all = _compute_IOU(neg_rois, gt_label)
        max_iou = np.max(iou_all, axis=1)
        iidx2 = np.where(max_iou < 0.15)[0]
        neg_patch_rois = neg_patch_rois[iidx2, :]
        neg_rois = neg_rois[iidx2, :]

        return neg_patch_rois, neg_rois, iidx1[iidx2].tolist()
        #
        # # 2. remove patches containing positive rois
        # overlaps = _compute_overlap(neg_patch_rois, gt_label[:, 1:])
        # is_good = np.ones(neg_patch_rois.shape[0])
        # for i, (nroi, overlap) in enumerate(zip(neg_patch_rois, overlaps)):
        #     oidx = np.where(overlap > 0.75)[0]
        #     if oidx.size == 0:
        #         continue
        #     gt_cand = gt_label[oidx, 1:]
        #     iou_gt = _compute_IOU_eltwise(_build_centered_rois(nroi, gt_cand), gt_cand)
        #     if np.max(iou_gt) > 0.35:
        #         is_good[i] = 0
        # iidx2 = np.where(is_good == 1)[0]
        # neg_patch_rois = neg_patch_rois[iidx2, :]
        #
        # return neg_patch_rois, iidx1[iidx2].tolist()

    def _debug_save_patches(self):
        import sys, os
        import cv2
        path_debug = '/home/hyunjoon/github/additions_mxnet/ssd_face/debug_patch/'
        path_pos = path_debug + 'pos'
        path_neg = path_debug + 'neg'
        if not os.path.exists(path_pos):
            os.mkdir(path_pos)
        if not os.path.exists(path_neg):
            os.mkdir(path_neg)

        curr_im_path = ''
        for i in range(self.patch_labels.shape[0]):
            im_path = self.image_path_from_index(i)
            label = self.label_from_index(i)
            if im_path != curr_im_path:
                with open(im_path, 'rb') as fh:
                    img_content = fh.read()
                img = mx.img.imdecode(img_content)
                curr_im_path = im_path
                if label[0] != 1.0:
                    hh = int(np.round(img.shape[0] * label[0]))
                    ww = int(np.round(img.shape[1] * label[0]))
                    img = mx.img.imresize(img, ww, hh, interp=2)
                img = img.asnumpy()
            try:
                patch = crop_roi_patch(img, np.array(label[6:]).astype(int)).asnumpy()
            except:
                import ipdb
                ipdb.set_trace()
            patch = patch[:, :, ::-1]
            roi = np.maximum(0, np.minimum(192, np.round(np.array(label[2:6]) * 192.0).astype(int)))
            patch_draw = patch.copy()
            cv2.rectangle(patch_draw, (roi[0], roi[1]), (roi[2], roi[3]), color=(0, 0, 255))

            if label[1] == 1:
                fn = os.path.join(path_pos, '{}.jpg'.format(i))
            else:
                fn = os.path.join(path_neg, '{}.jpg'.format(i))
            cv2.imwrite(fn, patch_draw)

def _draw_random_trans_patches(roi, min_iou, max_iou, patch_shape):
    min_trans_range = (1.0 - max_iou) / (2.0 * (1.0 + max_iou))
    max_trans_range = (1.0 - min_iou) / (2.0 * (1.0 + min_iou))
    scale_range = max_trans_range

    n_sample = roi.shape[0]
    samples = np.random.uniform(-1, 1, (n_sample, 2))
    r = samples[:, 0:1] * (max_trans_range - min_trans_range) + min_trans_range
    t = samples[:, 1:2] * np.pi
    x = r * np.cos(t) * roi[:, 2:3]
    y = r * np.sin(t) * roi[:, 3:4]
    cx = roi[:, 0:1] + (roi[:, 2:3] - 1.0) / 2.0
    cy = roi[:, 1:2] + (roi[:, 3:4] - 1.0) / 2.0
    ll = cx + x - (patch_shape - 1.0) / 2.0
    uu = cy + y - (patch_shape - 1.0) / 2.0

    patch_rois = np.zeros((n_sample, 4))
    patch_rois[:, 0:1] = ll
    patch_rois[:, 1:2] = uu
    patch_rois[:, 2:] = patch_shape

    sample_rois = roi.copy()
    sample_rois[:, 0:1] = roi[:, 0:1] + x
    sample_rois[:, 1:2] = roi[:, 1:2] + y
    return patch_rois, sample_rois

def _build_centered_rois(lhs, rhs):
    cx = lhs[0] + (lhs[2]-1.0) / 2.0
    cy = lhs[1] + (lhs[3]-1.0) / 2.0

    n_rhs = rhs.shape[0]
    ww2 = (rhs[:, 2:3]-1.0) / 2.0
    hh2 = (rhs[:, 3:4]-1.0) / 2.0
    rois = np.hstack((np.round(cx-ww2), np.round(cy-hh2), rhs[:, 2:3], rhs[:, 3:4]))
    return rois

def _compute_IOU_eltwise(lhs, rhs):
    iou_x = _compute_IOU_1d_eltwise(lhs[:, 0], lhs[:, 0]+lhs[:, 2], rhs[:, 0], rhs[:, 0]+rhs[:, 2])
    iou_y = _compute_IOU_1d_eltwise(lhs[:, 1], lhs[:, 1]+lhs[:, 3], rhs[:, 1], rhs[:, 1]+rhs[:, 3])
    return iou_x * iou_y

def _compute_IOU_1d_eltwise(p0, p1, q0, q1):
    ''' p0, p1, q0, q1: size of (n_rows, ) '''
    nr_p = p0.size
    nr_q = q1.size
    assert nr_p == nr_q
    min_ = np.maximum(p0, q0)
    max_ = np.minimum(p1, q1)
    i_ = np.maximum(0, max_ - min_)
    u_ =  (p1 - p0) * (q1 - q0) - i_
    return i_ / np.maximum(u_, 0.000001)

def _compute_IOU(lhs, rhs):
    iou_x = _compute_IOU_1d(lhs[:, 0], lhs[:, 0]+lhs[:, 2], rhs[:, 0], rhs[:, 0]+rhs[:, 2])
    iou_y = _compute_IOU_1d(lhs[:, 1], lhs[:, 1]+lhs[:, 3], rhs[:, 1], rhs[:, 1]+rhs[:, 3])
    return iou_x * iou_y

def _compute_IOU_1d(p0, p1, q0, q1):
    ''' p0, p1, q0, q1: size of (n_rows, ) '''
    nr_p = p0.size
    nr_q = q1.size
    min_ = np.maximum( \
            np.tile(np.reshape(p0, (-1, 1)), (1, nr_q)), \
            np.tile(np.reshape(q0, (1, -1)), (nr_p, 1)) )
    max_ = np.minimum( \
            np.tile(np.reshape(p1, (-1, 1)), (1, nr_q)), \
            np.tile(np.reshape(q1, (1, -1)), (nr_p, 1)) )
    i_ = np.maximum(max_ - min_, 0.0)

    u_ = np.tile(np.reshape(p1-p0, (-1, 1)), (1, nr_q)) + \
            np.tile(np.reshape(q1-q0, (1, -1)), (nr_p, 1)) 
    return i_ / np.maximum(0.000001, u_ - i_)

def _compute_overlap(lhs, rhs):
    ''' 
    return a matrix size of (n_lhs, n_rhs) 
    lhs is generally bigger than rhs
    lhs: (n_lhs, 4), each row contains (xmin, ymin, width, height)
    rhs: (n_rhs, 4), same as lhs
    '''
    iou_x = _compute_overlap_1d(lhs[:, 0], lhs[:, 0]+lhs[:, 2], rhs[:, 0], rhs[:, 0]+rhs[:, 2])
    iou_y = _compute_overlap_1d(lhs[:, 1], lhs[:, 1]+lhs[:, 3], rhs[:, 1], rhs[:, 1]+rhs[:, 3])
    return iou_x * iou_y

def _compute_overlap_1d(p0, p1, q0, q1):
    ''' p0, p1, q0, q1: size of (n_rows, ) '''
    nr_p = p0.size
    nr_q = q1.size
    min_ = np.maximum( \
            np.tile(np.reshape(p0, (-1, 1)), (1, nr_q)), \
            np.tile(np.reshape(q0, (1, -1)), (nr_p, 1)) )
    max_ = np.minimum( \
            np.tile(np.reshape(p1, (-1, 1)), (1, nr_q)), \
            np.tile(np.reshape(q1, (1, -1)), (nr_p, 1)) )
    i_ = np.maximum(max_ - min_, 0.0)

    u_ = np.tile(np.reshape(q1-q0, (1, -1)), (nr_p, 1)) 
    return np.minimum(1.0, i_ / u_)

# for test
if __name__ == '__main__':
    devkit_path = '/home/hyunjoon/fd/joint_cascade/data/wider'
    image_set = 'val'
    shuffle=True
    is_train = True

    WiderPatch(devkit_path=devkit_path, image_set=image_set, shuffle=shuffle, is_train=is_train)

