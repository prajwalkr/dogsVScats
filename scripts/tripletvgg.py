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
from random import randint, choice, sample
from sys import setrecursionlimit, argv
import keras.backend as K
import theano as T
from utils import dumper, resizer, kaggleTest, visualizer, segTest

ROOT = dirname(dirname(abspath(__file__)))
TRAIN_DIR, VAL_DIR = ROOT + '/train', ROOT + '/validation'
weights_path = ROOT + '/vgg16_weights.h5'
num_cats_train = len(listdir(TRAIN_DIR + '/cats'))
num_dogs_train = len(listdir(TRAIN_DIR + '/dogs'))
num_cats_val = len(listdir(VAL_DIR + '/cats'))
num_dogs_val = len(listdir(VAL_DIR + '/dogs'))
samples_per_epoch = num_cats_train + num_dogs_train

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

def tripletLoss(y_true,anchor):
	alpha = 0.2
	ones,zeros = anchor/anchor, 0*anchor
	anchor2 = T.tensor.concatenate((zeros,-2*anchor),axis=1)
	first_part = y_true*T.tensor.concatenate((ones,zeros),axis=1)
	second_part = y_true*anchor2
	return K.abs(K.sum(first_part + second_part) + alpha)

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
	fccmodel = Sequential()
	
	fccmodel.add(Flatten(input_shape=CNNmodel.layers[-1].output_shape[1:]))
	'''model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
	model.add(Dropout(0.5))'''
	fccmodel.add(Dense(256, activation='relu', W_constraint=maxnorm(3)))
	fccmodel.add(Dropout(0.5))
	fccmodel.add(Dense(64, activation='tanh'))

	CNNmodel.add(fccmodel)

	return CNNmodel

def init_model(preload=None):
	'''if preload:
		return load_model(preload)'''

	global model
	model = VGG_16()
	
	if preload:
		model.load_weights(preload)

def reshape(x):
	return x.reshape(1, channels, img_width, img_height)

def tripletDataGen():
	global model
	train_datagen = ImageDataGenerator(rotation_range=45,
	width_shift_range=0.2, height_shift_range=0.2,
	zoom_range=0.2, horizontal_flip=True).flow_from_directory(
	TRAIN_DIR, target_size=(img_width, img_height),
	batch_size=3, class_mode='binary')

	while 1:
		inputs, outputs = [], []
		while len(inputs) < mini_batch_sz:
			X, pred = train_datagen.next()
			pred = list(pred)
			if pred.count(1.) < 1 or pred.count(0.) < 1:
				continue
			oneindices = [i for i,val in enumerate(pred) if val == 1]
			zeroindices = [i for i,val in enumerate(pred) if val == 0]
			if len(oneindices) == 1:
				negative = X[oneindices[0]]
				positive_index, anchor_index = sample(zeroindices, 2)
			else:
				negative_index = sample(zeroindices, 1)
				negative = X[negative_index]
				positive_index, anchor_index = sample(oneindices, 2)
			positive, anchor = X[positive_index], X[anchor_index]
			pos_embedding = model.predict(reshape(positive))
			neg_embedding = model.predict(reshape(negative))
			anc_embedding = model.predict(reshape(anchor))
			y = np.concatenate((np.square(pos_embedding) - np.square(neg_embedding), 
										pos_embedding - neg_embedding),axis=1)
			inputs.append(anchor)
			outputs.append(y.reshape(y.shape[1]))
		yield (np.asarray(inputs),np.asarray(outputs))
		
def runner(epochs):
	print 'Model training begins..'
	datagen = tripletDataGen()
	global model
	model.compile(optimizer=SGD(1e-2, decay= 1e-4, momentum=0.9), loss=tripletLoss)
	#checkpoint = ModelCheckpoint('current.h5','val_loss',1,True)
	print 'Model compiled.'
	try:
		model.fit_generator(datagen, samples_per_epoch, epochs, verbose=1)
	except Exception as e:
		print e
	finally:
		fname = dumper(model,'cnn')
		print 'Model saved to disk at {}'.format(fname)
		return model
	print 'Model trained.'

def main(preload):
	global model
	if preload == 'none': preload = None
	init_model(preload=preload)
	runner(epochs=100)

if __name__ == '__main__':
	main(argv[1])