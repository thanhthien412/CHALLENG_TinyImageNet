from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam,SGD
from keras.models import load_model
import argparse
import json
import config
from datagenerator import Generator
from preprocessdata import Meansubtraction
import h5py
from model import GoogleNetMini
import matplotlib.pyplot as plt
import keras
import numpy as np
import tensorflow as tf
import keras.backend as K

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str,
help="path to *specific* model checkpoint to load")
ap.add_argument("-s", "--epoch", type=int, default=20,
help="epoch to restart training at")
ap.add_argument("-se", "--start", type=int, default=0,
help="epoch to restart training at")


args = vars(ap.parse_args())

class CustomSaver(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        # or save after some epoch, each k-th epoch etc.
        self.model.save(config.MODEL_PATH.format((epoch+1+args['start'])))

physical_devices = tf.config.list_physical_devices('GPU')
print('device: ',physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

aug = ImageDataGenerator(rotation_range=18, zoom_range=0.15,
width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
horizontal_flip=True, fill_mode="nearest")

means = json.loads(open(config.DATASET_MEAN).read())

meanprocess=Meansubtraction(means['R'],means['G'],means['B'])

trainGen = Generator(config.TRAIN_HDF5,config.NUM_CLASSES, 64,[meanprocess],
                     aug=aug)
valGen = Generator(config.VAL_HDF5,config.NUM_CLASSES, 64,[meanprocess],
                     aug=aug)
if (args['model'] is None):
    print("[INFO]: TRAINING")
    model=GoogleNetMini.build(64,64,3,config.NUM_CLASSES,0.0002)
    opt=Adam(1e-3)
    # opt=SGD(1e-2,0.9)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
            metrics=["accuracy"])
else:
    print("[INFO]: LOADING MODEL")
    model=load_model(args['model'])
    K.set_value(model.optimizer.lr, 1e-5)

saver = CustomSaver()
history=model.fit_generator(trainGen.generator(),
                    steps_per_epoch=trainGen.numImages//64,
                    validation_data=valGen.generator(),
                    validation_steps=valGen.numImages//64,
                    epochs=args['epoch'],
                    callbacks=[saver],
                    max_queue_size=10,
                    verbose=1,
                    shuffle=True,
                    )

trainGen.close()
valGen.close()

print(history.history.keys())

N = np.arange(0, len(history.history["loss"]))
plt.style.use("ggplot")
plt.figure()
plt.plot(N, history.history["loss"], label="train_loss")
plt.plot(N, history.history["val_loss"], label="val_loss")
plt.plot(N, history.history["accuracy"], label="train_acc")
plt.plot(N, history.history["val_accuracy"], label="val_acc")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(config.GRAPH_PATH.format(args['epoch']))
plt.close()