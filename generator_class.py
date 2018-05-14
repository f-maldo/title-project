from keras.utils import np_utils
import numpy as np
from skimage import io
from skimage.transform import resize


class GeneratorClass(object):
    def __init__(self, dim_x=224, dim_y=224, batch_size=16, shuffle=True):
        """Constructor

        :param dim_x: Size of x axe for generator
        :param dim_y: Size of y axe for generator
        :param batch_size: Size of batch
        :param shuffle: For shuffle selection of batches
        """
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.batch_size = batch_size
        self.shuffle = shuffle

    def generate(self, paths, batch_size=16):
        """Generate data and labels for Keras

        :param paths: Path to files for generation
        :param batch_size: Size of the batch
        """
        self.batch_size = batch_size
        while 1:
            indexes = self.__get_exploration_order(paths)
            imax = int(len(indexes) / self.batch_size)
            for i in range(imax):
                paths_temp = [paths[k] for k in indexes[i * self.batch_size:(i + 1) * self.batch_size]]
                x, y = self.__data_generation(paths_temp)
                yield x, y

    def __get_exploration_order(self, list_paths):
        """Get indexes of files

        :param list_paths: List with paths
        :return: Number of indexes
        """
        indexes = np.arange(len(list_paths))
        if self.shuffle is True:
            np.random.shuffle(indexes)
        return indexes

    def __data_generation(self, paths_temp):
        """Load data and labels

        :param paths_temp: Specific paths for the batch
        :return: Data and binary class
        """
        x = np.empty((self.batch_size, 3, self.dim_y, self.dim_x))
        y = np.empty(self.batch_size, dtype=int)
        for i, path in enumerate(paths_temp):
            image_temp = io.imread(path[0])
            image_temp = resize(image_temp, (self.dim_y, self.dim_x))
            x[i, :, :, :] = image_temp.transpose((2, 0, 1))
            y[i] = int(path[1])
        return x, self.sparsify(y)

    @staticmethod
    def sparsify(y):
        """Convert a class to binary form

        :param y: Original class
        :return: Binary class
        """
        n_classes = 163
        return np_utils.to_categorical(y, n_classes)
