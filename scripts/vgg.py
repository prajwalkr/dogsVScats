from keras.models import Sequential, load_model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.constraints import maxnorm
from keras.optimizers import RMSprop, SGD
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
from os.path import dirname, abspath
from os import listdir
import numpy as np
import h5py, pickle
from os.path import dirname, abspath
from scipy import ndimage
from random import randint, choice
from sys import setrecursionlimit, argv

from utils import dumper, resizer, tester, kaggleTest


ROOT = dirname(dirname(abspath(__file__)))
weights_path = ROOT + '/vgg16_weights.h5'
channels, img_width, img_height = 3, 64, 64
mini_batch_sz = 16

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

    CNNmodel.add(ZeroPadding2D((1, 1)))
    CNNmodel.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1', trainable=False))
    CNNmodel.add(ZeroPadding2D((1, 1)))
    CNNmodel.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2', trainable=False))
    CNNmodel.add(MaxPooling2D((2, 2), strides=(2, 2)))

    CNNmodel.add(ZeroPadding2D((1, 1)))
    CNNmodel.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1', trainable=False))
    CNNmodel.add(ZeroPadding2D((1, 1)))
    CNNmodel.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2', trainable=False))
    CNNmodel.add(ZeroPadding2D((1, 1)))
    CNNmodel.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3', trainable=False))
    CNNmodel.add(MaxPooling2D((2, 2), strides=(2, 2)))

    CNNmodel.add(ZeroPadding2D((1, 1)))
    CNNmodel.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1', trainable=False))
    CNNmodel.add(ZeroPadding2D((1, 1)))
    CNNmodel.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2', trainable=False))
    CNNmodel.add(ZeroPadding2D((1, 1)))
    CNNmodel.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3', trainable=False))
    CNNmodel.add(MaxPooling2D((2, 2), strides=(2, 2)))

    CNNmodel.add(ZeroPadding2D((1, 1)))
    CNNmodel.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1', trainable=False))
    CNNmodel.add(ZeroPadding2D((1, 1)))
    CNNmodel.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2', trainable=False))
    CNNmodel.add(ZeroPadding2D((1, 1)))
    CNNmodel.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3', trainable=False))
    CNNmodel.add(MaxPooling2D((2, 2), strides=(2, 2)))

    CNNmodel = weight_loader(CNNmodel)
    return CNNmodel

def init_model(preload=None):
    '''if preload:
        return VGG_16(), load_model(preload)'''

    vgg = VGG_16()
    model = Sequential()
    
    model.add(Flatten(input_shape=vgg.layers[-1].output_shape[1:]))
    model.add(Dense(1024, activation='relu', W_constraint=maxnorm(3), W_regularizer=l2()))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu', W_constraint=maxnorm(3), W_regularizer=l2()))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    if preload:
        model.load_weights(preload)
    return vgg, model

def DataAugmenter(X):
    for i in xrange(len(X)):
        #### rotation
        if choice([True,False]):
            X[i] = ndimage.rotate(X[i], randint(-15,15),reshape=False)

        #### translation
        if choice([True,False]):
            X[i] = ndimage.shift(X[i], randint(-15,15))
    
    return X

def DataGen():
    path = dirname(dirname(abspath(__file__))) + '/train.h5'
    with h5py.File(path) as hf:
        cats, dogs = hf.get('data')

    global validation_data, samples_per_epoch, cnnModel
    cat_train, cat_val = np.split(cats,[int(0.8*len(cats))])
    dog_train, dog_val = np.split(dogs, [int(0.8*len(dogs))])

    val_x = np.concatenate((cat_val, dog_val))
    val_x = resizer(val_x, (len(val_x), channels, img_height, img_width))
    val_y = np.concatenate((np.zeros(len(cat_val),dtype=np.float32),
                            np.ones(len(dog_val),dtype=np.float32)))
    del cat_val, dog_val
    validation_data = (cnnModel.predict(val_x, batch_size=16, verbose=1), val_y)
    del val_x, val_y

    samples_per_epoch = mini_batch_sz*((len(cat_train) + len(dog_train))/mini_batch_sz)
    y = np.concatenate((np.zeros(len(cat_train),dtype=np.float32),
                        np.ones(len(dog_train),dtype=np.float32)))

    num_cat, num_dog = mini_batch_sz >> 1, mini_batch_sz >> 1
    y = np.concatenate((np.zeros(num_cat,dtype=np.float32),
                        np.ones(num_dog,dtype=np.float32)))
    while 1:
        indices = np.random.randint(len(cat_train), size=num_cat)
        cat_samples = cat_train[indices]
        indices = np.random.randint(len(cat_train), size=num_dog)
        dog_samples = dog_train[indices]
        X = np.concatenate((cat_samples, dog_samples))
        del dog_samples, cat_samples
        X = resizer(X, (len(X), channels, img_height, img_width))
        X = DataAugmenter(X)
        X = cnnModel.predict(X, batch_size=len(X) >> 1)
        yield X,y

def runner(model, epochs):
    global samples_per_epoch, validation_data
    training_data = DataGen()
    training_data.next()
    model.compile(optimizer=RMSprop(lr=1e-5), loss='binary_crossentropy')
    checkpoint = ModelCheckpoint('current.h5','val_loss',1,True)
    print 'Model compiled.'
    try:
        model.fit_generator(training_data,samples_per_epoch,epochs,
                        verbose=1,validation_data=validation_data,callbacks=[checkpoint])
    except Exception as e:
        print e
    finally:
        fname = dumper(model,'cnn')
        print 'Model saved to disk at {}'.format(fname)
        return model

def main(preload='../models/2016-10-25-07:40:34.384867-cnn.h5', mode='train'):
    global cnnModel
    cnnModel, model = init_model(preload)
    if mode == 'test':
        return tester(model, cnnModel)
    if mode == 'kaggle':
        return kaggleTest(model, cnnModel)
    return runner(model, 5000)

if __name__ == '__main__':
    main(mode=argv[1] if len(argv) else 'train')