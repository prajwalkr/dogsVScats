from keras.models import Sequential, load_model, Model
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU, ELU
from keras.layers import Input, merge, GlobalAveragePooling2D
from keras.constraints import maxnorm
from keras.optimizers import RMSprop, SGD, Adam
from keras.regularizers import l2, activity_l2
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from os.path import dirname, abspath
from os import listdir
import numpy as np
import h5py, pickle
from scipy import ndimage
from random import randint, choice, shuffle, sample
from sys import setrecursionlimit, argv
from utils import dumper, kaggleTest, visualizer
from utils import *
from vgg import VGG_16 as vgg
from resnet import ResNet50
from spaced_rep import spacedRunner
from squeezenet import SqueezeNet

ROOT = dirname(dirname(abspath(__file__)))
TRAIN_DIR, VAL_DIR = ROOT + '/train', ROOT + '/validation'
num_cats_train = len(listdir(TRAIN_DIR + '/cats'))
num_dogs_train = len(listdir(TRAIN_DIR + '/dogs'))
num_cats_val = len(listdir(VAL_DIR + '/cats'))
num_dogs_val = len(listdir(VAL_DIR + '/dogs'))
samples_per_epoch = num_cats_train + num_dogs_train
nb_val_samples = num_cats_val + num_dogs_val

channels, img_width, img_height = 3, 300, 300
mini_batch_sz = 16

use_multicrop = False
use_multiscale = False
w1, h1 = 320, 320
w2, h2 = 240, 240
w3, h3 = 400, 400
num_filters, num_scales = 4, 3
# w3, h3 = 240, 320
# w4, h4 = 320, 240
# w5, h5 = 320, 400
# w6, h6 = 400, 320

def multiscale_model(preload=None):

	shared_conv_1 = Convolution2D(num_filters, 3, 3, activation = "linear")
	shared_conv_2 = Convolution2D(num_filters, 3, 3, activation = "linear")
	
	input_1 = Input(shape=(channels, w1, h1))
	zero_pad_1 = ZeroPadding2D((1, 1)) (input_1)
	conved_1 = shared_conv_1 (zero_pad_1)
	elu_1 = ELU() (conved_1)
	zero_pad_2 = ZeroPadding2D((1, 1)) (elu_1)
	conved_2 = shared_conv_2 (zero_pad_2)
	elu_2 = ELU() (conved_2)
	pool_1 = MaxPooling2D(pool_size=(4,4), strides=(4,4)) (elu_2)

	input_2 = Input(shape=(channels, w2, h2))
	zero_pad_1 = ZeroPadding2D((1, 1)) (input_2)
	conved_1 = shared_conv_1 (zero_pad_1)
	elu_1 = ELU() (conved_1)
	zero_pad_2 = ZeroPadding2D((1, 1)) (elu_1)
	conved_2 = shared_conv_2 (zero_pad_2)
	elu_2 = ELU() (conved_2)
	pool_2 = MaxPooling2D(pool_size=(3,3), strides=(3,3)) (elu_2)

	input_3 = Input(shape=(channels, w3, h3))
	zero_pad_1 = ZeroPadding2D((1, 1)) (input_3)
	conved_1 = shared_conv_1 (zero_pad_1)
	elu_1 = ELU() (conved_1)
	zero_pad_2 = ZeroPadding2D((1, 1)) (elu_1)
	conved_2 = shared_conv_2 (zero_pad_2)
	elu_2 = ELU() (conved_2)
	pool_3 = MaxPooling2D(pool_size=(5,5), strides=(5,5)) (elu_2)

	multiscaleInputBlock = merge([pool_1, pool_2, pool_3], mode='concat', concat_axis=1)

	zero_pad_1 = ZeroPadding2D((1, 1)) (multiscaleInputBlock)
	conved_1 = Convolution2D(64, 3, 3, activation='linear') (zero_pad_1)
	elu_1 = ELU() (conved_1)
	zero_pad_2 = ZeroPadding2D((1, 1)) (elu_1)
	conved_2 = Convolution2D(64, 3, 3, activation='linear') (zero_pad_2)
	elu_2 = ELU() (conved_2)
	zero_pad_3 = ZeroPadding2D((1, 1)) (elu_2)
	conved_3 = Convolution2D(64, 3, 3, activation='linear') (zero_pad_3)
	elu_3 = ELU() (conved_3)
	dropout = Dropout(0.5) (elu_3)
	pool = MaxPooling2D(pool_size=(5,5)) (dropout)
	
	zero_pad_1 = ZeroPadding2D((1, 1)) (pool)
	conved_1 = Convolution2D(128, 3, 3, activation='linear') (zero_pad_1)
	elu_1 = ELU() (conved_1)
	zero_pad_2 = ZeroPadding2D((1, 1)) (elu_1)
	conved_2 = Convolution2D(128, 3, 3, activation='linear') (zero_pad_2)
	elu_2 = ELU() (conved_2)
	zero_pad_3 = ZeroPadding2D((1, 1)) (elu_2)
	conved_3 = Convolution2D(128, 3, 3, activation='linear') (zero_pad_3)
	elu_3 = ELU() (conved_3)
	dropout = Dropout(0.5) (elu_3)
	pool = MaxPooling2D(pool_size=(3,3)) (dropout)

	flat = Flatten() (pool)
	dense_1 = Dense(1024, activation="linear") (flat)
	elu_1 = ELU() (dense_1)
	dropout_1 = Dropout(0.5) (elu_1)
	output = Dense(1, activation="sigmoid") (dropout_1)

	model = Model(input=[input_1, input_2, input_3], output=output)

	if preload:
		model.load_weights(preload)

	return model

def multicrop_model(preload=None):
	shared_conv_1 = Convolution2D(num_filters, 3, 3, activation = "linear")
	shared_conv_2 = Convolution2D(num_filters, 3, 3, activation = "linear")
	
	input_1 = Input(shape=(channels, w1, h1))
	zero_pad_1 = ZeroPadding2D((1, 1)) (input_1)
	conved_1 = shared_conv_1 (zero_pad_1)
	elu_1 = ELU() (conved_1)
	zero_pad_2 = ZeroPadding2D((1, 1)) (elu_1)
	conved_2 = shared_conv_2 (zero_pad_2)
	elu_2 = ELU() (conved_2)
	pool_1 = MaxPooling2D(pool_size=(4,4), strides=(4,4)) (elu_2)

	input_2 = Input(shape=(channels, w2, h2))
	zero_pad_1 = ZeroPadding2D((1, 1)) (input_2)
	conved_1 = shared_conv_1 (zero_pad_1)
	elu_1 = ELU() (conved_1)
	zero_pad_2 = ZeroPadding2D((1, 1)) (elu_1)
	conved_2 = shared_conv_2 (zero_pad_2)
	elu_2 = ELU() (conved_2)
	pool_2 = MaxPooling2D(pool_size=(3,3), strides=(3,3)) (elu_2)

	input_3 = Input(shape=(channels, w3, h3))
	zero_pad_1 = ZeroPadding2D((1, 1)) (input_3)
	conved_1 = shared_conv_1 (zero_pad_1)
	elu_1 = ELU() (conved_1)
	zero_pad_2 = ZeroPadding2D((1, 1)) (elu_1)
	conved_2 = shared_conv_2 (zero_pad_2)
	elu_2 = ELU() (conved_2)
	pool_3 = MaxPooling2D(pool_size=(5,5), strides=(5,5)) (elu_2)

	multiscalecropInputBlock = merge([pool_1, pool_2, pool_3], mode='concat', concat_axis=1)

	zero_pad_1 = ZeroPadding2D((1, 1)) (multiscalecropInputBlock)
	conved_1 = Convolution2D(32, 3, 3, activation='linear') (zero_pad_1)
	elu_1 = ELU() (conved_1)
	zero_pad_2 = ZeroPadding2D((1, 1)) (elu_1)
	conved_2 = Convolution2D(32, 3, 3, activation='linear') (zero_pad_2)
	elu_2 = ELU() (conved_2)
	pool = MaxPooling2D(pool_size=(5,5)) (elu_2)
	
	zero_pad_1 = ZeroPadding2D((1, 1)) (pool)
	conved_1 = Convolution2D(64, 3, 3, activation='linear') (zero_pad_1)
	elu_1 = ELU() (conved_1)
	zero_pad_2 = ZeroPadding2D((1, 1)) (elu_1)
	conved_2 = Convolution2D(64, 3, 3, activation='linear') (zero_pad_2)
	elu_2 = ELU() (conved_2)
	zero_pad_3 = ZeroPadding2D((1, 1)) (elu_2)
	conved_3 = Convolution2D(64, 3, 3, activation='linear') (zero_pad_3)
	elu_3 = ELU() (conved_3)
	pool = MaxPooling2D(pool_size=(3,3)) (elu_3)

	# zero_pad_1 = ZeroPadding2D((1, 1)) (pool)
	# conved_1 = Convolution2D(128, 3, 3, activation='linear') (zero_pad_1)
	# elu_1 = ELU() (conved_1)
	# zero_pad_2 = ZeroPadding2D((1, 1)) (elu_1)
	# conved_2 = Convolution2D(128, 3, 3, activation='linear') (zero_pad_2)
	# elu_2 = ELU() (conved_2)
	# zero_pad_3 = ZeroPadding2D((1, 1)) (elu_2)
	# conved_3 = Convolution2D(128, 3, 3, activation='linear') (zero_pad_3)
	# elu_3 = ELU() (conved_3)
	# pool = MaxPooling2D(pool_size=(2,2)) (elu_3)

	flat = Flatten() (pool)
	dense_1 = Dense(512, activation="linear") (flat)
	elu_1 = ELU() (dense_1)
	dropout_1 = Dropout(0.5) (elu_1)
	output = Dense(1, activation="sigmoid") (dropout_1)

	model = Model(input=[input_1, input_2, input_3], output=output)

	if preload:
		model.load_weights(preload)

	return model

def init_model(preload=None, declare=False, use_vgg=False, use_resnet=True):
	if use_multiscale and use_vgg and use_squeezenet: raise ValueError('Incorrect params')
	if not declare and preload: return load_model(preload)
	if use_multiscale: return multiscale_model(preload)
	if use_multicrop: return multicrop_model(preload)
	
	if use_resnet: 
		if not preload:
			weights_path = ROOT + '/resnet50_tf_notop.h5'
			body = ResNet50(input_shape=(img_width, img_width, channels), weights_path=weights_path)
		for layer in body.layers: layer.trainable = False

		head = body.output
		batchnormed = BatchNormalization(axis=3)(head)
		avgpooled = GlobalAveragePooling2D()(batchnormed)
		dropout = Dropout(0.5) (avgpooled)
		output = Dense(1, activation="sigmoid")(dropout)
		model = Model(body.input, output)

		if preload: model.load_weights(preload)
		return model

	if use_vgg:
		model = vgg()
		model.add(Dropout(0.5))
		model.add(Flatten())
		model.add(Dense(64, activation = 'linear'))
		model.add(ELU())
		model.add(Dropout(0.5))
		model.add(Dense(1, activation='sigmoid'))

		if preload:
			model.load_weights(preload)

		return model

	model = Sequential()
	model.add(ZeroPadding2D((1, 1), input_shape=(channels, img_width, img_height)))
	model.add(Convolution2D(16, 3, 3, activation = "linear"))
	model.add(ELU())
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(16, 3, 3, activation = "linear"))
	model.add(ELU())
	model.add(MaxPooling2D(pool_size=(5,5)))
	model.add(Dropout(0.5))

	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(32, 3, 3, activation = "linear"))
	model.add(ELU())
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(32, 3, 3, activation = "linear"))
	model.add(ELU())
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(32, 3, 3, activation = "linear"))
	model.add(ELU())
	model.add(MaxPooling2D(pool_size=(3,3)))
	model.add(Dropout(0.5))

	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(64, 3, 3, activation = "linear"))
	model.add(ELU())
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(64, 3, 3, activation = "linear"))
	model.add(ELU())
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(64, 3, 3, activation = "linear"))
	model.add(ELU())
	model.add(MaxPooling2D(pool_size=(3,3)))
	model.add(Dropout(0.5))

	model.add(Flatten())
	model.add(Dense(64, activation='linear'))
	model.add(ELU())
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))

	if preload: model.load_weights(preload)
	return model

def standardized(gen, training=False, vgg=True):
	MEAN_VALUE = np.array([103.939, 116.779, 123.68])
	mean, stddev = pickle.load(open('meanSTDDEV'))
	while 1:
		X,y = gen.next()
		for i in xrange(len(X)):

			if training:
				pass
				# if randint(0, 4)//4:
				# 	X[i] = blur(X[i])
				# if randint(0, 4)//4:
				# 	X[i] = random_bright_shift(X[i])
				# if randint(0, 4)//4:
				# 	X[i] = random_contrast_shift(X[i])

			if vgg:
				for j in xrange(3): X[i][j] -= MEAN_VALUE[j]
				X[i] = X[i][::-1]
			else:
				X[i] = (X[i] - mean) / stddev
		yield X,y

def submean(X, ms):
	for i in xrange(len(X)): X[i] = (X[i] - ms[0]) / ms[1]
	return X

def ms_traingen():
	train_datagen = ImageDataGenerator(rotation_range=30.,
		horizontal_flip=True, fill_mode='reflect').flow_from_directory(
		TRAIN_DIR,
		target_size=(max([w1,w2,w3]), max([h1,h2,h3])),
		batch_size=mini_batch_sz,
		class_mode='binary'
		)
	meanstdev = [pickle.load(open('meanSTDDEV320')), pickle.load(open('meanSTDDEV240')),
			pickle.load(open('meanSTDDEV400'))]

	while 1:
		X,y = train_datagen.next()
		for i in xrange(len(X)):	
			if randint(0, 4)//4:
				X[i] = random_bright_shift(X[i])
			if randint(0, 4)//4:
				X[i] = random_contrast_shift(X[i])

		quad1, quad2 = sample(np.random.permutation(4),2)
		x1, y1 = getXY(quad1, w1)
		x2, y2 = getXY(quad2, w2, imsize=w1)
		X1 = submean(cropX(X, x=x1, y=y1, size=w1), meanstdev[0])
		X2 = submean(cropX(resizeX(X, w1), x=x2, y=y2, size=w2), meanstdev[1])
		X3 = submean(X, meanstdev[2])

		yield ([X1, X2, X3], y)

def ms_valgen():
	validation_datagen = ImageDataGenerator().flow_from_directory(
		VAL_DIR,
		target_size=(max([w1,w2,w3]), max([h1,h2,h3])),
		batch_size=mini_batch_sz,
		class_mode='binary'
		)
	meanstdev = [pickle.load(open('meanSTDDEV320')), pickle.load(open('meanSTDDEV240')),
			pickle.load(open('meanSTDDEV400'))]

	while 1:
		X,y = validation_datagen.next()
		quad1, quad2 = sample(np.random.permutation(4),2)
		x1, y1 = getXY(quad1, w1)
		x2, y2 = getXY(quad2, w2, imsize=w1)
		X1 = submean(cropX(X, x=x1, y=y1, size=w1), meanstdev[0])
		X2 = submean(cropX(resizeX(X, w1), x=x2, y=y2, size=w2), meanstdev[1])
		X3 = submean(X, meanstdev[2])

		yield ([X1, X2, X3], y)

def resnet_sub_mean(gen):
	while 1:
		X,y = gen.next()
		for i in xrange(len(X)): X[i] = (X[i] - np.array([103.939, 116.779, 123.68]))[...,::-1]
		yield X,y

def DataGen():
	train_datagen = ImageDataGenerator(rotation_range=5., width_shift_range=0.05,
		height_shift_range=0.05, channel_shift_range=10., horizontal_flip=True, fill_mode='constant')

	validation_datagen = ImageDataGenerator(horizontal_flip=True)

	train_generator = train_datagen.flow_from_directory(
		TRAIN_DIR,
		target_size=(img_width, img_height),
		batch_size=mini_batch_sz,
		class_mode='binary')

	validation_generator = validation_datagen.flow_from_directory(
		VAL_DIR,target_size=(img_width, img_height),
		batch_size=mini_batch_sz,
		class_mode='binary')

	return train_generator, validation_generator

def runner(model, epochs):
	initial_LR = 1e-4
	if not use_multiscale and not use_multicrop: training_gen, val_gen = DataGen()
	else: training_gen, val_gen = ms_traingen(), ms_valgen()

	model.compile(optimizer=Adam(initial_LR), loss='binary_crossentropy')

	val_checkpoint = ModelCheckpoint('bestval.h5','val_loss',1, True)
	cur_checkpoint = ModelCheckpoint('current.h5')
	def lrForEpoch(i): return initial_LR if i < 2 else initial_LR / (i / 2)
	lrScheduler = LearningRateScheduler(lrForEpoch)
	print 'Model compiled.'

	try:
		model.fit_generator(training_gen,samples_per_epoch,epochs,
						verbose=1,validation_data=val_gen,nb_val_samples=nb_val_samples,
						callbacks=[val_checkpoint, cur_checkpoint, lrScheduler])
	except Exception as e:
		print e
	finally:
		fname = dumper(model,'cnn')
		print 'Model saved to disk at {}'.format(fname)
		return model

def save_failed(model):
	_, v = DataGen()
	saved = 50
	# MEAN_VALUE = np.array([103.939, 116.779, 123.68])
	mean, stddev = pickle.load(open('meanSTDDEV'))
	done = 0
	# cat_pred, dog_pred = [],[]
	yt, yp = [], []
	while 1:
		X, y_true = v.next()
		y_pred = model.predict(X)
		for i, pred in enumerate(y_pred):
			# if np.abs(y_pred[0] - y_true[0]) > 0.6:
			# 	X[i] = (X[i] * stddev) + mean
			# 	write_image(X[i], '../failures/{}.jpg'.format(randint(0,1000000000)))
			# 	saved -= 1
			# if y_true[i] < 0.5: cat_pred.append(pred[0])
			# else: dog_pred.append(pred[0])
			if pred[0] < 0.3:
				yp.append(0.)
			if pred[0] > 0.7:
				yp.append(1.)
			else:
				yp.append(0.7)
			yt.append(y_true[i])

		if done % 100 == 0: print done
		done += len(X)
		if done >= 5716: break
		if saved <= 0:
			break
	print logloss(y_true, y_pred) / 5716
	print logloss([0.] * len(cat_pred), cat_pred), logloss([1.] * len(dog_pred), dog_pred)
	# print float(sum(cat_pred) + sum(dog_pred)) / 5716.

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
	if mode == 'failed':
		return save_failed(model)
	if mode == 'train':
		return runner(model, 100)
	else: raise ValueError('Incorrect mode')

if __name__ == '__main__':
	main(argv[1:])
