import fire
from imghdr import what
from keras.models import load_model
import numpy as np
from os.path import isfile
from scipy.io import loadmat
from skimage import io
from skimage.transform import resize


class PredictCar(object):
    def __init__(self):
        """Constructor class

        """
        self.img_rows = 224
        self.img_cols = 224

    def predict(self, path_to_file):
        """Method for predict brand car
        Command line example: python3 eval.py predict --path_to_file="path/to/file.jpg"


        :param path_to_file: Path to image file in jpg format
        :return: True if the prediction is done, False if happens some error
        """
        if isfile(path_to_file) is False:
            print("Path is not a file or file don't exist")
            return False
        if what(path_to_file) is not "jpeg":
            print("Only enter jpg images")
            return False

        # Load Keras model
        model = load_model("cache/final_model.h5")

        # Load make model
        make_model = loadmat("cache/make_model_name.mat")["make_names"]

        im = io.imread(path_to_file)
        im = resize(im, (self.img_rows, self.img_cols))
        im = im.reshape((1, 3, self.img_rows, self.img_cols))
        prediction = model.predict(im)
        arg = np.argmax(prediction) - 1
        make_model_prediction = make_model[arg][0]
        print("The car brand is " + str(make_model_prediction))
        return True

if __name__ == "__main__":
    fire.Fire(PredictCar)
