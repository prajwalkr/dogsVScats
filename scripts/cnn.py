from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, ZeroPadding2D
from keras.models import Sequential, load_model
from keras.layers.pooling import MaxPooling2D
from keras.constraints import maxnorm
from keras.optimizers import RMSprop, SGD
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
from os.path import dirname, abspath
from os import listdir
from utils import dumper
import h5py
import numpy as np
from builder import prep_data
from scipy import ndimage
from random import randint, choice
import cv2
from utils import resizer
from scripts.alex import alex
from scripts.vgg import VGG_16 as vgg
import pickle
from sys import setrecursionlimit

ROOT = dirname(dirname(abspath(__file__)))

def kaggleTest(model):
	TEST_DIR = dirname(dirname(abspath(__file__))) + '/test/'
	fnames = [TEST_DIR + fname for fname in listdir(TEST_DIR)]
	X = prep_data(fnames)
	X = resizer(X)
	ids = [x[:-4] for x in [fname for fname in listdir(TEST_DIR)]]
	y = model.predict(X,verbose=1)
	with open('out.csv','w') as f:
		f.write('id,label\n')
		for i,pred in zip(ids,y):
			f.write('{},{}\n'.format(i,str(pred[0])))
	
def init_model(preload=None,load_weights=False,compileModel=False):
	print 'Building model...'

	
	if preload:
		return load_model(preload)

	model = Sequential()

	model.add(Convolution2D(nb_filter=32,nb_row=3,nb_col=3,
							input_shape=(3, 128, 128)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D())

	model.add(Convolution2D(nb_filter=32,nb_row=2,nb_col=2))
	model.add(Activation('relu'))
	model.add(MaxPooling2D())

	model.add(Convolution2D(64,2,2))
	model.add(Activation('relu'))
	model.add(MaxPooling2D())

	model.add(Convolution2D(64,2,2))
	model.add(Activation('relu'))
	model.add(MaxPooling2D())

	model.add(Convolution2D(128,2,2))
	model.add(Activation('relu'))

	model.add(Flatten())
	model.add(Dense(1024,W_constraint=maxnorm(3),W_regularizer=l2(0.01)))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	model.add(Dense(512,W_constraint=maxnorm(3),W_regularizer=l2(0.01)))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	model.add(Dense(2, activation='sigmoid'))
	model.add(Activation('softmax'))

	if load_weights:
		model.load_weights(load_weights)

	if compileModel:
		model.compile(optimizer='adam', loss='binary_crossentropy')
		print 'Model compiled.'

	return model

def DataAugmenter(X):
	X = resizer(X)
	for i in xrange(len(X)):
		#### rotation
		if choice([True,False]):
			X[i] = ndimage.rotate(X[i], randint(-5,5),reshape=False)

		#### translation
		if choice([True,False]):
			X[i] = ndimage.shift(X[i], randint(-5,5))
	
	return X

def shuffle(X,y):
	pairs = zip(X,y)
	np.random.shuffle(pairs)
	for i, (img, target) in enumerate(pairs):
		X[i], y[i] = img, target
	return X,y

def DataGen():
	path = dirname(dirname(abspath(__file__))) + '/train.h5'
	with h5py.File(path) as hf:
		cats, dogs = hf.get('data')

	cats, dogs = cats - np.mean(cats), dogs - np.mean(dogs)
	global validation_data, samples_per_epoch
	cat_train, cat_val = np.split(cats,[int(0.8*len(cats))])
	dog_train, dog_val = np.split(dogs, [int(0.8*len(dogs))])

	val_x = resizer(np.concatenate((cat_val, dog_val)))
	val_y = np.concatenate((np.array([[1,0]]*len(cat_val),dtype=np.float32),
							np.array([[0,1]]*len(dog_val),dtype=np.float32)))
	#validation_data = (val_x, val_y)
	cat_train,_ = np.split(cat_train, [10])
	dog_train,_ = np.split(dog_train, [10])
	validation_data = (resizer(np.concatenate((cat_train, dog_train))), 
						np.concatenate((np.array([[1,0]]*len(cat_train),dtype=np.float32),
							np.array([[0,1]]*len(dog_train),dtype=np.float32))))
	mini_batch_sz = 20
	split = 10
	cat,dog = split,mini_batch_sz - split
	samples_per_epoch = mini_batch_sz*((len(cat_train) + len(dog_train))/mini_batch_sz)

	y = np.concatenate((np.array([[1,0]]*len(cat_train),dtype=np.float32),
							np.array([[0,1]]*len(dog_train),dtype=np.float32)))
	while 1:
		indices = np.random.randint(len(cat_train),size=cat)
		cat_samples = cat_train[indices]
		indices = np.random.randint(len(cat_train),size=dog)
		dog_samples = dog_train[indices]
		X = np.concatenate((cat_samples, dog_samples))
		X = resizer(X)
		yield shuffle(X,y)

def runner(model, epochs):
	global samples_per_epoch, cat_val, dog_val
	training_data = DataGen()
	training_data.next()

	model.compile(optimizer='adam', loss='binary_crossentropy')
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

def main(model):
	return runner(model, 5000)

if __name__ == '__main__':
	main()