from keras.models import Sequential, load_model
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.constraints import maxnorm
from keras.optimizers import RMSprop, SGD
from keras.regularizers import l2, activity_l2
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from os.path import dirname, abspath
from os import listdir
import numpy as np
import h5py, pickle
from os.path import dirname, abspath
from scipy import ndimage
from random import randint, choice
from sys import setrecursionlimit, argv

from utils import dumper, resizer, kaggleTest, visualizer, segTest


ROOT = dirname(dirname(abspath(__file__)))
TRAIN_DIR, VAL_DIR = ROOT + '/train', ROOT + '/validation'
num_cats_train = len(listdir(TRAIN_DIR + '/cats'))
num_dogs_train = len(listdir(TRAIN_DIR + '/dogs'))
num_cats_val = len(listdir(VAL_DIR + '/cats'))
num_dogs_val = len(listdir(VAL_DIR + '/dogs'))
samples_per_epoch = num_cats_train + num_dogs_train
nb_val_samples = num_cats_val + num_dogs_val

channels, img_width, img_height = 3, 150, 150
mini_batch_sz = 4

def init_model(preload=None):
    '''if preload:
        return load_model(preload)'''

    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(channels, img_width, img_height)))
    model.add(Convolution2D(32, 3, 3, activation = "linear"))
    model.add(BatchNormalization(axis=1))
    model.add(PReLU())
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(32, 3, 3, activation = "linear"))#, W_regularizer=l2(0.0001)))
    model.add(BatchNormalization(axis=1))
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(3,3))) #if image is 150x150
    ###
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation = "linear"))#, W_regularizer=l2(0.0001)))
    model.add(BatchNormalization(axis=1))
    model.add(PReLU())
    model.add(Convolution2D(64, 3, 3, activation = "linear"))#, W_regularizer=l2(0.0001)))
    model.add(BatchNormalization(axis=1))
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(2,2)))
    ###
    model.add(Convolution2D(128, 3, 3, activation = "linear"))#, W_regularizer=l2(0.0001)))
    model.add(BatchNormalization(axis=1))
    model.add(PReLU())
    model.add(Convolution2D(128, 3, 3, activation = "linear"))#, W_regularizer=l2(0.0001)))
    model.add(BatchNormalization(axis=1))
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(2,2)))
    ###
    model.add(Convolution2D(256, 3, 3, activation = "linear"))#, W_regularizer=l2(0.0001)))
    model.add(BatchNormalization(axis=1))
    model.add(PReLU())
    model.add(Convolution2D(256, 3, 3, activation = "linear"))#, W_regularizer=l2(0.0001)))
    model.add(BatchNormalization(axis=1))
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(6,6)))

    # MLP
    model.add(Flatten())
    model.add(Dense(1024, activation="linear"))#, W_regularizer=l2(0.0001)))
    model.add(BatchNormalization())
    model.add(PReLU())
    #model.add(Dropout(p=0.4))
    model.add(Dense(512, activation="linear"))#, W_regularizer=l2(0.0001)))
    model.add(BatchNormalization())
    model.add(PReLU())
    #model.add(Dropout(p=0.4))
    model.add(Dense(2, activation="linear"))#, W_regularizer=l2(0.0001)))
    model.add(BatchNormalization())
    model.add(Activation("softmax"))

    if preload:
        model.load_weights(preload)

    return model

def DataGen():
    train_datagen = ImageDataGenerator(rotation_range=45,
    width_shift_range=0.2, height_shift_range=0.2, channel_shift_range=10.,
    zoom_range=0.2, horizontal_flip=True,vertical_flip=True)

    validation_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(img_width, img_height),
        batch_size=mini_batch_sz,
        class_mode='categorical')

    validation_generator = validation_datagen.flow_from_directory(
        VAL_DIR,target_size=(img_width, img_height),
        batch_size=mini_batch_sz,
        class_mode='categorical')

    return train_generator, validation_generator

def runner(model, epochs):
    global validation_data
    training_gen, val_gen = DataGen()

    model.compile(optimizer=SGD(5e-3, decay=1e-5, momentum=0.9, nesterov=True), loss='categorical_crossentropy')
    checkpoint = ModelCheckpoint('current.h5','val_loss',1,True)
    print 'Model compiled.'
    try:
        model.fit_generator(training_gen,samples_per_epoch,epochs,
                        verbose=1,validation_data=val_gen,nb_val_samples=nb_val_samples,
                        callbacks=[checkpoint])
    except Exception as e:
        print e
    finally:
        fname = dumper(model,'cnn')
        print 'Model saved to disk at {}'.format(fname)
        return model

def main(args):
    mode, preload = args
    if preload == 'none': preload = None
    model = init_model(preload)
    if mode == 'test':
        return tester(model)
    if mode == 'kaggle':
        return kaggleTest(model)
    if mode == 'vis':
        return visualizer(model)
    return runner(model, 100)

if __name__ == '__main__':
    main(argv[1:])
