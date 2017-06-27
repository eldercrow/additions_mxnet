from imdb import Imdb
import random
import os

class ConcatPatchDB(Imdb):
    """
    ConcatDB is used to concatenate multiple imdbs to form a larger db.
    It is very useful to combine multiple dataset with same classes.
    Parameters
    ----------
    imdbs : Imdb or list of Imdb
        Imdbs to be concatenated
    shuffle : bool
        whether to shuffle the initial list
    """
    def __init__(self, imdbs, merge_classes=False, shuffle=False):
        super(ConcatPatchDB, self).__init__('concatdb')
        assert shuffle == False, "Shuffle should be always False for patch DB."
        if not isinstance(imdbs, list):
            imdbs = [imdbs]
        self.imdbs = imdbs
        self.merge_classes = merge_classes
        self._check_classes()
        if merge_classes:
            self._merge_class_names()
        self.image_set_index = self._load_image_set_index(shuffle)
        self.full_image_set_index = self._load_full_image_set_index(shuffle)
        # self.pad_labels()

    def _check_classes(self):
        """
        check input imdbs, make sure they have same classes
        """
        try:
            self.classes = self.imdbs[0].classes
            self.num_classes = len(self.classes)
        except AttributeError:
            # fine, if no classes is provided
            pass

        if self.num_classes > 0:
            for db in self.imdbs:
                assert db.classes[0] == '__background__', "The first class should be '__background__'."
                if self.merge_classes == False:
                    assert self.classes == db.classes, "Multiple imdb must have same classes"

    def _merge_class_names(self):
        '''
        '''
        all_classes = []
        for db in self.imdbs:
            all_classes += db.classes[1:] # without '__background__'

        # get unique elements
        all_classes = list(set(all_classes))
        all_classes.insert(0, '__background__')

        self.clsname2idx = []
        for db in self.imdbs:
            clsname2idx = []
            for cls in db.classes:
                clsname2idx.append(all_classes.index(cls))
            self.clsname2idx.append(clsname2idx)
        self.classes = all_classes

    def _load_image_set_index(self, shuffle):
        '''
        '''
        self.num_images = 0
        for db in self.imdbs:
            self.num_images += db.patch_labels.shape[0]
        indices = range(self.num_images)
        if shuffle:
            random.shuffle(indices)
        return indices

    def _load_full_image_set_index(self, shuffle):
        """
        get total number of images, init indices

        Parameters
        ----------
        shuffle : bool
            whether to shuffle the initial indices
        """
        self.num_orig_images = 0
        for db in self.imdbs:
            self.num_orig_images += db.num_orig_images
        indices = range(self.num_orig_images)
        if shuffle:
            random.shuffle(indices)
        return indices

    def full_image_path_from_index(self, index):
        """
        given image index, find out full path

        Parameters
        ----------
        index: int
            index of a specific image

        Returns
        ----------
        full path of this image
        """
        assert self.full_image_set_index is not None, "Dataset not initialized"
        n_db, n_index = self._locate_full_index(index)
        return self.imdbs[n_db].full_image_path_from_index(n_index)

    def image_path_from_index(self, index):
        '''
        given patch index, find out full path of the image
        containng the patche
        '''
        assert self.image_set_index is not None, "Dataset not initialized"
        n_db, n_index = self._locate_index(index)
        return self.imdbs[n_db].image_path_from_index(n_index)

    def img_shape_from_index(self, index):
        assert self.image_set_index is not None, "Dataset not initialized"
        n_db, n_index = self._locate_index(index)
        return self.imdbs[n_db].img_shape_from_index(n_index)

    def label_from_index(self, index):
        """
        given image index, return preprocessed ground-truth

        Parameters
        ----------
        index: int
            index of a specific image

        Returns
        ----------
        ground-truths of this image
        """
        assert self.image_set_index is not None, "Dataset not initialized"
        pos = self.image_set_index[index]
        n_db, n_index = self._locate_index(index)
        label = self.imdbs[n_db].label_from_index(n_index)
        if self.merge_classes == True:
            import ipdb
            ipdb.set_trace()
            for i, l in enumerate(label):
                label[i][1] = self.clsname2idx[n_db][l[1]]
        return self.imdbs[n_db].label_from_index(n_index)

    def _locate_full_index(self, index):
        """
        given index, find out sub-db and sub-index

        Parameters
        ----------
        index : int
            index of a specific image

        Returns
        ----------
        a tuple (sub-db, sub-index)
        """
        assert index >= 0 and index < self.num_orig_images, "index out of range"
        pos = self.full_image_set_index[index]
        for k, v in enumerate(self.imdbs):
            if pos >= v.num_orig_images:
                pos -= v.num_orig_images
            else:
                return (k, pos)

    def _locate_index(self, index):
        """
        given index, find out sub-db and sub-index

        Parameters
        ----------
        index : int
            index of a specific image

        Returns
        ----------
        a tuple (sub-db, sub-index)
        """
        assert index >= 0 and index < self.num_images, "index out of range"
        pos = self.image_set_index[index]
        for k, v in enumerate(self.imdbs):
            if pos >= v.num_images:
                pos -= v.num_images
            else:
                return (k, pos)


    def reset_patch(self):
        for db in self.imdbs:
            db.reset_patch()
