from keras.models import Sequential, load_model
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
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

from utils import dumper, kaggleTest, visualizer


ROOT = dirname(dirname(abspath(__file__)))
TRAIN_DIR, VAL_DIR = ROOT + '/train', ROOT + '/validation'
weights_path = ROOT + '/vgg16_weights.h5'
num_cats_train = len(listdir(TRAIN_DIR + '/cats'))
num_dogs_train = len(listdir(TRAIN_DIR + '/dogs'))
num_cats_val = len(listdir(VAL_DIR + '/cats'))
num_dogs_val = len(listdir(VAL_DIR + '/dogs'))
samples_per_epoch = num_cats_train + num_dogs_train
nb_val_samples = num_cats_val + num_dogs_val

channels, img_width, img_height = 3, 224, 224
mini_batch_sz = 4

def weight_loader(cnnmodel):
    with h5py.File(weights_path) as f:
        for k in range(f.attrs['nb_layers']):
            if k >= len(cnnmodel.layers):
                # we don't look at the last (fully-connected) layers in the savefile
                break
            g = f['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
            cnnmodel.layers[k].set_weights(weights)
    return cnnmodel

def VGG_16():
    CNNmodel = Sequential()

    CNNmodel.add(ZeroPadding2D((1, 1), input_shape=(channels, img_width, img_height)))
    CNNmodel.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1', trainable=False))
    CNNmodel.add(ZeroPadding2D((1, 1)))
    CNNmodel.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2', trainable=False))
    CNNmodel.add(MaxPooling2D((2, 2), strides=(2, 2)))
    #CNNmodel.add(Dropout(0.1))

    CNNmodel.add(ZeroPadding2D((1, 1)))
    CNNmodel.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1', trainable=False))
    CNNmodel.add(ZeroPadding2D((1, 1)))
    CNNmodel.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2', trainable=False))
    CNNmodel.add(MaxPooling2D((2, 2), strides=(2, 2)))
    #CNNmodel.add(Dropout(0.2))

    CNNmodel.add(ZeroPadding2D((1, 1)))
    CNNmodel.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1', trainable=False))
    CNNmodel.add(ZeroPadding2D((1, 1)))
    CNNmodel.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2', trainable=False))
    CNNmodel.add(ZeroPadding2D((1, 1)))
    CNNmodel.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3', trainable=False))
    CNNmodel.add(MaxPooling2D((2, 2), strides=(2, 2)))
    #CNNmodel.add(Dropout(0.3))

    CNNmodel.add(ZeroPadding2D((1, 1)))
    CNNmodel.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1', trainable=False))
    CNNmodel.add(ZeroPadding2D((1, 1)))
    CNNmodel.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2', trainable=False))
    CNNmodel.add(ZeroPadding2D((1, 1)))
    CNNmodel.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3', trainable=False))
    CNNmodel.add(MaxPooling2D((2, 2), strides=(2, 2)))
    #CNNmodel.add(Dropout(0.3))

    CNNmodel.add(ZeroPadding2D((1, 1)))
    CNNmodel.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1', trainable=False))
    CNNmodel.add(ZeroPadding2D((1, 1)))
    CNNmodel.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2', trainable=False))
    CNNmodel.add(ZeroPadding2D((1, 1)))
    CNNmodel.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3', trainable=False))
    CNNmodel.add(MaxPooling2D((2, 2), strides=(2, 2)))
    #CNNmodel.add(Dropout(0.5))

    CNNmodel = weight_loader(CNNmodel)
    return CNNmodel

def init_model(preload=None):
    if preload:
        return load_model(preload)

    if preload == 'custom':
        return custom()

    vgg = VGG_16()
    
    if preload:
        vgg.load_weights(preload)

    return vgg

def sub_mean(img):
    img = img[::-1]
    means = [103.939, 116.779, 123.68]
    for i in xrange(3):
        img[i] -= means[i]
    return img

def customgen(gen):
    while 1:
        X,y = gen.next()
        for i in xrange(len(X)):
            X[i] = sub_mean(X[i])
        yield X, y

def DataGen():
    '''train_datagen = ImageDataGenerator(rotation_range=45,
    width_shift_range=0.2, height_shift_range=0.2, channel_shift_range=20.,
    zoom_range=0.2, horizontal_flip=True)'''
    train_datagen = ImageDataGenerator(rotation_range=45, horizontal_flip=True)

    validation_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(img_width, img_height),
        batch_size=mini_batch_sz,
        class_mode='binary')

    validation_generator = validation_datagen.flow_from_directory(
        VAL_DIR,target_size=(img_width, img_height),
        batch_size=mini_batch_sz,
        class_mode='binary')

    return customgen(train_generator), customgen(validation_generator)

def runner(model, epochs):
    global validation_data
    training_gen, val_gen = DataGen()

    model.compile(optimizer=SGD(5e-5, momentum=0.9, nesterov=True), loss='binary_crossentropy')
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
