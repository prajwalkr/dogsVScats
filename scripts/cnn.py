from keras.models import Sequential, load_model
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU, ELU
from keras.constraints import maxnorm
from keras.optimizers import RMSprop, SGD
from keras.regularizers import l2, activity_l2
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from os.path import dirname, abspath
from os import listdir
import numpy as np
import h5py, pickle
from scipy import ndimage
from random import randint, choice
from sys import setrecursionlimit, argv
from utils import dumper, kaggleTest, visualizer
from utils import *
from vgg import VGG_16 as vgg

ROOT = dirname(dirname(abspath(__file__)))
TRAIN_DIR, VAL_DIR = ROOT + '/train', ROOT + '/validation'
num_cats_train = len(listdir(TRAIN_DIR + '/cats'))
num_dogs_train = len(listdir(TRAIN_DIR + '/dogs'))
num_cats_val = len(listdir(VAL_DIR + '/cats'))
num_dogs_val = len(listdir(VAL_DIR + '/dogs'))
samples_per_epoch = num_cats_train + num_dogs_train
nb_val_samples = num_cats_val + num_dogs_val

channels, img_width, img_height = 3, 224, 224
mini_batch_sz = 8

def other_one(preload=None):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(channels, img_width, img_height)))
    model.add(Convolution2D(32, 3, 3, activation = "linear"))
    model.add(BatchNormalization(axis=1))
    model.add(PReLU())
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(32, 3, 3, activation = "linear"))
    model.add(BatchNormalization(axis=1))
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(3,3)))
    
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation = "linear"))
    model.add(BatchNormalization(axis=1))
    model.add(PReLU())
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation = "linear"))
    model.add(BatchNormalization(axis=1))
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(3,3)))
    # model.add(Dropout(0.5))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation = "linear"))
    model.add(BatchNormalization(axis=1))
    model.add(PReLU())
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation = "linear"))
    model.add(BatchNormalization(axis=1))
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(0.5))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation = "linear"))
    model.add(BatchNormalization(axis=1))
    model.add(PReLU())
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation = "linear"))
    model.add(BatchNormalization(axis=1))
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(0.5))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation = "linear"))
    model.add(BatchNormalization(axis=1))
    model.add(PReLU())
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation = "linear"))
    model.add(BatchNormalization(axis=1))
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(2048, activation="linear"))
    model.add(BatchNormalization())
    model.add(PReLU())
    # model.add(Dropout(0.3))
    model.add(Dense(2048, activation="linear"))
    model.add(BatchNormalization())
    model.add(PReLU())
    # model.add(Dropout(0.3))
    model.add(Dense(2, activation="linear"))
    model.add(BatchNormalization())
    model.add(Activation("softmax"))

    if preload:
        model.load_weights(preload)
    return model

def init_model(preload=None, use_vgg=False):
    if use_vgg:
        CNNmodel = vgg()
        model = Sequential()
        model.add(Flatten(input_shape=CNNmodel.layers[-1].output_shape[1:]))
        model.add(Dense(512, activation="linear"))
        model.add(PReLU())
        model.add(Dropout(p=0.5))
        model.add(Dense(256, activation="linear"))
        model.add(PReLU())
        model.add(Dropout(p=0.5))
        model.add(Dense(2, activation="linear"))
        model.add(Activation("softmax"))
        CNNmodel.add(model)

        if preload:
            CNNmodel.load_weights(preload)
        return CNNmodel

    # return load_model(preload)
    return other_one(preload)

    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(channels, img_width, img_height)))
    model.add(Convolution2D(32, 3, 3, activation = "linear"))
    model.add(BatchNormalization(axis=1))
    model.add(ELU())
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(32, 3, 3, activation = "linear"))
    model.add(BatchNormalization(axis=1))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation = "linear"))
    model.add(BatchNormalization(axis=1))
    model.add(ELU())
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation = "linear"))
    model.add(BatchNormalization(axis=1))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation = "linear"))
    model.add(BatchNormalization(axis=1))
    model.add(ELU())
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation = "linear"))
    model.add(BatchNormalization(axis=1))
    model.add(ELU())
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation = "linear"))
    model.add(BatchNormalization(axis=1))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation = "linear"))
    model.add(BatchNormalization(axis=1))
    model.add(ELU())
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation = "linear"))
    model.add(BatchNormalization(axis=1))
    model.add(ELU())
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation = "linear"))
    model.add(BatchNormalization(axis=1))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation = "linear"))
    model.add(BatchNormalization(axis=1))
    model.add(ELU())
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation = "linear"))
    model.add(BatchNormalization(axis=1))
    model.add(ELU())
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation = "linear"))
    model.add(BatchNormalization(axis=1))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(1024, 3, 3, activation = "linear"))
    model.add(BatchNormalization(axis=1))
    model.add(ELU())
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(1024, 3, 3, activation = "linear"))
    model.add(BatchNormalization(axis=1))
    model.add(ELU())
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(1024, 3, 3, activation = "linear"))
    model.add(BatchNormalization(axis=1))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(2048, activation="linear"))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Dense(2048, activation="linear"))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Dense(2, activation="linear"))
    model.add(BatchNormalization())
    model.add(Activation("softmax"))

    if preload:
        model.load_weights(preload)

    return model

def customgen(traingen):
    # MEAN_VALUE = np.array([103.939, 116.779, 123.68])
    while 1:
        X,y = traingen.next()
        for i in xrange(len(X)):
            if randint(0, 8)//8:
                X[i] = random_bright_shift(X[i])
            if randint(0, 5)//5:
                X[i] = blur(X[i])
            # if randint(0, 7)//7:
            #     X[i] = random_contrast_shift(X[i])
            # X[i] = X[i][::-1]
            # for j in xrange(3):
            #     X[i][j] = X[i][j] - MEAN_VALUE[j]
        yield X,y 

def standardized(gen):
    # MEAN_VALUE = np.array([103.939, 116.779, 123.68])
    mean, stddev = pickle.load(open('meanSTDDEV'))
    while 1:
        X,y = gen.next()
        for i in xrange(len(X)):
            X[i] = (X[i] - mean) / stddev
        yield X,y

def addmean(X):
    MEAN_VALUE = [103.939, 116.779, 123.68]
    for i in xrange(len(X)):
            for j in xrange(3):
                    X[i][j] += MEAN_VALUE[j]
            X[i] = X[i][::-1]
    return X

def DataGen():
    train_datagen = ImageDataGenerator(horizontal_flip=True, channel_shift_range=5.,
        rotation_range=10., zoom_range=0.2, fill_mode="constant", cval=0.)

    validation_datagen = ImageDataGenerator(horizontal_flip=True)

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

    model.compile(optimizer=SGD(1e-4), loss='categorical_crossentropy')

    val_checkpoint = ModelCheckpoint('bestval.h5','val_loss',1,True)
    cur_checkpoint = ModelCheckpoint('current.h5')
    # csvlogger = CSVLogger('current.csv', append=True)
    print 'Model compiled.'

    try:
        model.fit_generator(training_gen,samples_per_epoch,epochs,
                        verbose=1,validation_data=val_gen,nb_val_samples=nb_val_samples,
                        callbacks=[val_checkpoint, cur_checkpoint])
    except Exception as e:
        print e
    finally:
        fname = dumper(model,'cnn')
        print 'Model saved to disk at {}'.format(fname)
        return model

def main(args):
    if len(args) == 2: mode, preload = args
    elif len(args) == 3: mode, preload, img_path = args
    else: raise ValueError('Only 2 or 3 args.')

    if preload == 'none': preload = None
    model = init_model(preload)
    if mode == 'test':
        return tester(model)
    if mode == 'kaggle':
        return kaggleTest(model)
    if mode == 'vis':
        return visualizer(model)
    if mode == 'activations':
        raise NotImplementedError('Prajwal is lazy')
    if mode == 'train':
        return runner(model, 100)
    else: raise ValueError('Incorrect mode')

if __name__ == '__main__':
    main(argv[1:])
