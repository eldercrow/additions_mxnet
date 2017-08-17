import os
import numpy as np
from imdb import Imdb


class OpenImage(Imdb):
    """
    Implementation of Imdb for OpenImage bounding box dataset

    Parameters:
    ----------
    anno_file : str
        annotation file for openimage, a .txt file
    image_dir : str
        image directory for coco images
    shuffle : bool
        whether initially shuffle image list

    """
    def __init__(self, anno_file, image_dir, shuffle=True, names='openimage.names'):
        assert os.path.isfile(anno_file), "Invalid annotation file: " + anno_file
        basename = os.path.splitext(os.path.basename(anno_file))[0]
        super(OpenImage, self).__init__('openimage_' + basename)
        self.image_dir = image_dir

        self.classes = self._load_class_names(names,
            os.path.join(os.path.dirname(__file__), 'names'))

        self.num_classes = len(self.classes)
        self._load_all(anno_file, shuffle)
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
        image_file = os.path.join(self.image_dir, name)
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
        with open(anno_file, 'r') as fh:
            for lstr in fh:
                linfo = lstr.strip().split('\t')
                filename = linfo[0]
                objs = np.reshape(np.array(linfo[1:]), (-1, 5))
                for i, o in enumerate(objs[:, 0]):
                    objs[i, 0] = self.classes.index(o)
                if objs.size:
                    labels.append(objs.astype(float))
                    image_set_index.append(filename)

        if shuffle:
            import random
            indices = range(len(image_set_index))
            random.shuffle(indices)
            image_set_index = [image_set_index[i] for i in indices]
            labels = [labels[i] for i in indices]
        # store the results
        self.image_set_index = image_set_index
        self.labels = labels


if __name__ == '__main__':
    anno_file = '/home/hyunjoon/dataset/openimage/rec/train_bb_annot.txt'
    image_dir = '/home/hyunjoon/dataset/openimage/train'

    OpenImage(anno_file, image_dir)
