import fire
from generator_class import GeneratorClass
from keras.layers import Dense, Conv2D, MaxPooling2D, ZeroPadding2D, Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np
from skimage import io
from skimage.transform import resize

from sklearn.metrics import log_loss


class FineTune(object):
    def start(self, batch=256, epochs=300):
        # Example to fine-tune on 3000 samples from Cifar10

        img_rows, img_cols = 224, 224
        channel = 3
        num_classes = 163
        batch_size = batch
        nb_epoch = epochs

        # Load our model
        model = self.vgg16_model(img_rows, img_cols, channel, num_classes)

        # Load data for generation
        x_train = np.load("cache/x_train.npy")
        y_train = np.load("cache/y_train.npy")

        train_paths = []
        for i, _x in enumerate(x_train):
            train_paths.append([x_train[i], y_train[i]])

        generator = GeneratorClass()
        training_generator = generator.generate(train_paths, batch_size)

        # Start Fine-tuning
        model.fit_generator(generator=training_generator,
                            steps_per_epoch=len(x_train) // batch_size,
                            epochs=nb_epoch,
                            verbose=1)

        model.save("cache/final_model2.h5")

    @staticmethod
    def vgg16_model(img_rows, img_cols, channel=1, num_classes=None):
        """VGG 16 Model for Keras
        Model Schema is based on
        https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
        ImageNet Pretrained Weights
        https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view?usp=sharing
        Parameters:
          img_rows, img_cols - resolution of inputs
          channel - 1 for grayscale, 3 for color
          num_classes - number of categories for our classification task
        """
        model = Sequential()
        model.add(ZeroPadding2D((1, 1), input_shape=(channel, img_rows, img_cols)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        # Add Fully Connected Layer
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1000, activation='softmax'))

        # Loads ImageNet pre-trained data
        model.load_weights('C:/Users/Felipe/Documents/Proyectos/machine-learning/final_project/cache/vgg16_weights_tf_dim_ordering_tf_kernels.h5')

        # Truncate and replace softmax layer for transfer learning
        model.layers.pop()
        model.outputs = [model.layers[-1].output]
        model.layers[-1].outbound_nodes = []
        model.add(Dense(num_classes, activation='softmax'))

        # Learning rate is changed to 0.001
        sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

        return model

if __name__ == '__main__':
    fire.Fire(FineTune)