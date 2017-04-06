from __future__ import print_function
import os
import numpy as np
from imdb import Imdb
import xml.etree.ElementTree as ET
# from evaluate.eval_voc import voc_eval
import cv2
import cPickle


class Wider(Imdb):
    """
    Implementation of Imdb for WIDER face dataset

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
    """
    IDX_VER = '170401_1' # for caching

    def __init__(self, image_set, devkit_path, shuffle=False, is_train=False):
        super(Wider, self).__init__('wider_' + image_set) # e.g. wider_trainval
        self.image_set = image_set
        # self.year = year
        self.devkit_path = devkit_path
        self.data_path = devkit_path # os.path.join(devkit_path, 'wider')
        self.extension = '.jpg'
        self.is_train = is_train

        self.classes = ['__background__', 'face',]
        # self.classes = ['aeroplane', 'bicycle', 'bird', 'boat',
        #                 'bottle', 'bus', 'car', 'cat', 'chair',
        #                 'cow', 'diningtable', 'dog', 'horse',
        #                 'motorbike', 'person', 'pottedplant',
        #                 'sheep', 'sofa', 'train', 'tvmonitor']

        self.config = { \
                'use_difficult': False, 
                'th_small': 8.0, 
                'comp_id': 'comp4', 
                'padding': 256 } 

        self.num_classes = len(self.classes)
        self.max_objects = 0

        # try to load cached data
        cached = self._load_from_cache()
        if cached is None: # no cached data, load from DB (and save)
            fn_cache = os.path.join(self.cache_path, self.name + '_' + self.IDX_VER + '.pkl')
            self.image_set_index = self._load_image_set_index(shuffle)
            self.num_images = len(self.image_set_index)
            if self.is_train:
                self.labels, self.max_objects = self._load_image_labels()
            self._save_to_cache()
        else:
            self.image_set_index = cached['image_set_index']
            self.num_images = len(self.image_set_index)
            if self.is_train:
                if 'labels' in cached:
                    self.labels = cached['labels']
                else:
                    self.labels, self.max_objects = self._load_image_labels()
                    self._save_to_cache()
        if shuffle:
            ridx = np.random.permutation(np.arange(self.num_images))
            image_set_index = [self.image_set_index[i] for i in ridx]
            labels = [self.labels[i] for i in ridx]
            self.image_set_index, self.labels = image_set_index, labels
        self._pad_labels()

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
                    if self.is_train:
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
            if self.is_train:
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

    def image_path_from_index(self, index):
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

    def label_from_index(self, index):
        """
        given image index, return preprocessed ground-truth

        Parameters:
        ----------
        index: int
            index of a specific image
        Returns:
        ----------
        ground-truths of this image
        """
        assert self.labels is not None, "Labels not processed"
        return self.labels[index, :, :]
        # label = self.labels[index]
        # assert label.shape[0] == self.padding, "Wider::label_from_index: Label shape does not match."
        # return label
        # return self.labels[index, :, :]

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
        temp = []
        max_objects = 0
        max_label = np.empty((0, 5))
        max_fn = ''

        # load ground-truth from xml annotations
        for idx in range(len(self.image_set_index)):
            bb_file, prop_file = self._label_path_from_index(idx)
            cls_id = self.classes.index('face')
            bbs = np.reshape(np.loadtxt(bb_file).astype(float), (-1, 4))
            if bbs.size == 0: 
                temp.append(np.empty((0, 5)))
                continue
            ww_img = 0
            hh_img = 0
            small_mask = np.minimum(bbs[:, 2], bbs[:, 3]) < self.config['th_small']
            # remove bbs that are 1) invalid or 2) too small and occluded.
            with open(prop_file, 'r') as fh:
                prop_data = fh.read().splitlines()
            for pdata in prop_data:
                prop_bb = pdata.split(' ')
                if prop_bb[0] == 'invalid_label_list':
                    invalid_mask = np.array(prop_bb[1:]).astype(int) == 1
                # if prop_bb[0] == 'occlusion_label_list':
                #     small_mask = np.logical_and(small_mask, np.array(prop_bb[1:]).astype(int) == 2)
                # also get image size
                if prop_bb[0] == 'image_size':
                    ww_img = int(prop_bb[1])
                    hh_img = int(prop_bb[2])
            if self.config['use_difficult'] is not True:
                invalid_mask = np.logical_or(invalid_mask, small_mask)
            # FOR DEBUG, save image size to .prop_label
            if ww_img == 0:
                fn_img = self.image_path_from_index(idx)
                img = cv2.imread(fn_img)
                prop_data.append('image_size %d %d' % (img.shape[1], img.shape[0]))
                with open(prop_file, 'w') as fh:
                    for pdata in prop_data:
                        fh.write(pdata + '\n')
                ww_img = img.shape[1]
                hh_img = img.shape[0]

            valid_idx = np.where(invalid_mask == False)[0]
            bbs = bbs[valid_idx, :]
            if bbs.size == 0: 
                temp.append(np.empty((0, 5)))
                continue

            # we need [xmin, ymin, xmax, ymax], but wider DB has [xmin, ymin, width, height]
            bbs[:, 2] += bbs[:, 0] 
            bbs[:, 3] += bbs[:, 1] 
            # normalize to [0, 1]
            bbs[:, 0::2] /= ww_img
            bbs[:, 1::2] /= hh_img

            bbs = np.minimum(np.maximum(bbs, 0.0), 1.0)

            label = np.zeros((bbs.shape[0], 5))
            label[:, 0] = cls_id
            label[:, 1:] = bbs
            temp.append(label)
            if label.shape[0] > max_objects:
                max_objects = label.shape[0]
                # max_label = label
                # max_fn = bb_file

        # add padding to labels so that the dimensions match in each batch
        # TODO: design a better way to handle label padding
        # [hyunjoon]: trying a better way to handle label padding for WIDER DB.
        assert max_objects > 0, "No objects found for any of the images"

        return temp, max_objects

        # assert max_objects > 0, "No objects found for any of the images"
        # assert max_objects <= self.config['padding'], \
        #         "# obj exceed padding, (%d > %d)" % (max_objects, self.config['padding'])
        # self.padding = self.config['padding']
        # labels = []
        # for label in temp:
        #     label = np.lib.pad(label, ((0, self.padding-label.shape[0]), (0,0)), \
        #                        'constant', constant_values=(-1, -1))
        #     labels.append(label)
        #
        # return np.array(labels)

    def _pad_labels(self):
        """ labels: list of ndarrays """
        self.padding = np.maximum(self.max_objects, self.config['padding'])
        for (i, label) in enumerate(self.labels):
            # # TODO: test, make roi square
            # cx = (label[:, 1] + label[:, 3]) / 2.0
            # cy = (label[:, 2] + label[:, 4]) / 2.0
            # max_roi_sz = np.maximum(label[:, 3] - label[:, 1], label[:, 4] - label[:, 2])
            # label[:, 1] = cx - max_roi_sz / 2.0
            # label[:, 2] = cy - max_roi_sz / 2.0
            # label[:, 3] = cx + max_roi_sz / 2.0
            # label[:, 4] = cy + max_roi_sz / 2.0
            padded = np.tile(np.full((5,), -1, dtype=np.float32), (self.padding, 1))
            padded[:label.shape[0], :] = label
            self.labels[i] = padded
            # if label.shape[0] < self.padding:
            #     self.labels[i] = np.lib.pad(label, ((0, self.padding-label.shape[0]), (0,0)), \
            #             'constant', constant_values=(-1, -1))
        self.labels = np.array(self.labels)

    # def evaluate_detections(self, detections):
    #     """
    #     top level evaluations
    #     Parameters:
    #     ----------
    #     detections: list
    #         result list, each entry is a matrix of detections
    #     Returns:
    #     ----------
    #         None
    #     """
    #     # make all these folders for results
    #     result_dir = os.path.join(self.devkit_path, 'results')
    #     if not os.path.exists(result_dir):
    #         os.mkdir(result_dir)
    #     year_folder = os.path.join(self.devkit_path, 'results', 'WIDER')
    #     if not os.path.exists(year_folder):
    #         os.mkdir(year_folder)
    #     res_file_folder = os.path.join(self.devkit_path, 'results', 'WIDER', 'Main')
    #     if not os.path.exists(res_file_folder):
    #         os.mkdir(res_file_folder)
    #
    #     self.write_pascal_results(detections)
    #     self.do_python_eval()
    #
    # def get_result_file_template(self):
    #     """
    #     this is a template
    #     VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
    #
    #     Returns:
    #     ----------
    #         a string template
    #     """
    #     res_file_folder = os.path.join(self.devkit_path, 'results', 'VOC' + self.year, 'Main')
    #     comp_id = self.config['comp_id']
    #     filename = comp_id + '_det_' + self.image_set + '_{:s}.txt'
    #     path = os.path.join(res_file_folder, filename)
    #     return path
    #
    # def write_pascal_results(self, all_boxes):
    #     """
    #     write results files in pascal devkit path
    #     Parameters:
    #     ----------
    #     all_boxes: list
    #         boxes to be processed [bbox, confidence]
    #     Returns:
    #     ----------
    #     None
    #     """
    #     for cls_ind, cls in enumerate(self.classes):
    #         print('Writing {} VOC results file'.format(cls))
    #         filename = self.get_result_file_template().format(cls)
    #         with open(filename, 'wt') as f:
    #             for im_ind, index in enumerate(self.image_set_index):
    #                 dets = all_boxes[im_ind]
    #                 if dets.shape[0] < 1:
    #                     continue
    #                 h, w = self._get_imsize(self.image_path_from_index(im_ind))
    #                 # the VOCdevkit expects 1-based indices
    #                 for k in range(dets.shape[0]):
    #                     if (int(dets[k, 0]) == cls_ind):
    #                         f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
    #                                 format(index, dets[k, 1],
    #                                        int(dets[k, 2] * w) + 1, int(dets[k, 3] * h) + 1,
    #                                        int(dets[k, 4] * w) + 1, int(dets[k, 5] * h) + 1))
    #
    # def do_python_eval(self):
    #     """
    #     python evaluation wrapper
    #
    #     Returns:
    #     ----------
    #     None
    #     """
    #     annopath = os.path.join(self.data_path, 'Annotations', '{:s}.xml')
    #     imageset_file = os.path.join(self.data_path, 'ImageSets', 'Main', self.image_set + '.txt')
    #     cache_dir = os.path.join(self.cache_path, self.name)
    #     aps = []
    #     # The PASCAL VOC metric changed in 2010
    #     use_07_metric = True if int(self.year) < 2010 else False
    #     print('VOC07 metric? ' + ('Y' if use_07_metric else 'No'))
    #     for cls_ind, cls in enumerate(self.classes):
    #         filename = self.get_result_file_template().format(cls)
    #         rec, prec, ap = voc_eval(filename, annopath, imageset_file, cls, cache_dir,
    #                                  ovthresh=0.5, use_07_metric=use_07_metric)
    #         aps += [ap]
    #         print('AP for {} = {:.4f}'.format(cls, ap))
    #     print('Mean AP = {:.4f}'.format(np.mean(aps)))
    #
    # def _get_imsize(self, im_name):
    #     """
    #     get image size info
    #     Returns:
    #     ----------
    #     tuple of (height, width)
    #     """
    #     img = cv2.imread(im_name)
    #     return (img.shape[0], img.shape[1])


# for test
if __name__ == '__main__':
    devkit_path = '/home/hyunjoon/fd/joint_cascade/data/wider'
    image_set = 'val'
    shuffle=True
    is_train = True

    Wider(devkit_path=devkit_path, image_set=image_set, shuffle=shuffle, is_train=is_train)
