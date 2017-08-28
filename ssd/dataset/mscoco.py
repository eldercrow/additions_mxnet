import os
import numpy as np
from imdb import Imdb
from pycocotools.coco import COCO


class Coco(Imdb):
    """
    Implementation of Imdb for MSCOCO dataset: https://http://mscoco.org

    Parameters:
    ----------
    anno_file : str
        annotation file for coco, a json file
    image_dir : str
        image directory for coco images
    shuffle : bool
        whether initially shuffle image list

    """
    IDX_VER = '170811_1'

    def __init__(self, anno_file, image_dir, shuffle=True, names='mscoco.names'):
        assert os.path.isfile(anno_file), "Invalid annotation file: " + anno_file
        basename = os.path.splitext(os.path.basename(anno_file))[0]
        super(Coco, self).__init__('coco_' + basename)
        self.image_dir = image_dir

        self.classes = self._load_class_names(names,
            os.path.join(os.path.dirname(__file__), 'names'))

        self.num_classes = len(self.classes)

        # try to load cached data
        cached = self._load_from_cache()
        if cached is None:  # no cached data, load from DB (and save)
            fn_cache = os.path.join(self.cache_path, self.name + '_' + self.IDX_VER + '.pkl')
            self._load_all(anno_file, shuffle)
            self._save_to_cache()
        else:
            self.image_set_index = cached['image_set_index']
            self.num_images = len(self.image_set_index)
            if self.is_train:
                self.labels = cached['labels']
        self.num_images = len(self.image_set_index)

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
        image_file = os.path.join(self.image_dir, 'images', name)
        assert os.path.isfile(image_file), 'Path does not exist: {}'.format(image_file)
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
        return self.labels[index]

    def _load_all(self, anno_file, shuffle):
        """
        initialize all entries given annotation json file

        Parameters:
        ----------
        anno_file: str
            annotation json file
        shuffle: bool
            whether to shuffle image list
        """
        image_set_index = []
        labels = []
        coco = COCO(anno_file)
        img_ids = coco.getImgIds()
        max_objects == 0
        for img_id in img_ids:
            # filename
            image_info = coco.loadImgs(img_id)[0]
            filename = image_info["file_name"]
            subdir = filename.split('_')[1]
            height = image_info["height"]
            width = image_info["width"]
            # label
            anno_ids = coco.getAnnIds(imgIds=img_id)
            annos = coco.loadAnns(anno_ids)
            label = []
            for anno in annos:
                cat_id = int(anno["category_id"])
                bbox = anno["bbox"]
                assert len(bbox) == 4
                xmin = float(bbox[0]) / width
                ymin = float(bbox[1]) / height
                xmax = xmin + float(bbox[2]) / width
                ymax = ymin + float(bbox[3]) / height
                label.append([cat_id, xmin, ymin, xmax, ymax, 0])
            if label:
                max_objects = max(max_objects, len(labels))
                labels.append(np.array(label))
                image_set_index.append(os.path.join(subdir, filename))

        assert max_objects > 0

        if shuffle:
            import random
            indices = range(len(image_set_index))
            random.shuffle(indices)
            image_set_index = [image_set_index[i] for i in indices]
            labels = [labels[i] for i in indices]
        # store the results
        self.image_set_index = image_set_index
        self.labels = labels
        self.max_objects = max_objects

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
        return cached

    def _save_to_cache(self):
        fn_cache = os.path.join(self.cache_path, self.name + '_' + self.IDX_VER + '.pkl')
        with open(fn_cache, 'wb') as fh:
            header = {'ver': self.IDX_VER, 'max_objects': self.max_objects}
            cPickle.dump(header, fh, cPickle.HIGHEST_PROTOCOL)
            cPickle.dump({'image_set_index': self.image_set_index}, fh, cPickle.HIGHEST_PROTOCOL)
            if self.is_train:
                cPickle.dump({'labels': self.labels}, fh, cPickle.HIGHEST_PROTOCOL)
