import fire
from keras.applications import VGG16
from keras import models, optimizers, layers
from keras.preprocessing.image import ImageDataGenerator


class CompCars(object):
    def __init__(self, height=100, width=150, batch_size=100,
                 train_dir='ImagesForFlow/train',
                 test_dir='ImagesForFlow/test'):
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.train_dir = train_dir
        self.test_dir = test_dir

    def train(self):
        vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(height, width, 3))
        vgg_conv.trainable = True
        set_trainable = False
        for layer in vgg_conv.layers:
            if layer.name == 'block5_conv1':
                set_trainable = True
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False

        model = models.Sequential()
        model.add(vgg_conv)
        model.add(layers.Flatten())
        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(163, activation='softmax'))

        train_datagen = ImageDataGenerator()
        validation_datagen = ImageDataGenerator()

        val_batchsize = int(self.batch_size / 10)

        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(self.height, self.width),
            batch_size=self.batch_size,
            class_mode='categorical')

        validation_generator = validation_datagen.flow_from_directory(
            self.test_dir,
            target_size=(self.height, self.width),
            batch_size=val_batchsize,
            class_mode='categorical',
            shuffle=False)

        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.RMSprop(lr=1e-4),
                      metrics=['acc'])

        model.fit_generator(
            train_generator,
            steps_per_epoch=train_generator.samples / train_generator.batch_size,
            epochs=30,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples / validation_generator.batch_size,
            verbose=1)

        model.save('Models/mark1.h5')


if __name__ == "__main__":
    fire.Fire(CompCars)
