from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential, load_model
from keras.layers.pooling import MaxPooling2D
from keras.constraints import maxnorm
from os.path import dirname, abspath
from datetime import datetime
import h5py
import numpy as np

def dumper(model,kind,fname=None):
	if not fname:
		fname = '{}/models/{}-{}.h5'.format(dirname(abspath(__file__)),
										str(datetime.now()).replace(' ','-'),kind)
	try:
		with open(fname,'w') as f:
			model.save(fname)
	except IOError:
		raise IOError('Unable to open: {}'.format(fname))
	return fname

def init_model(preload=None):
	print 'Building model...'

	if preload:
		return load_model(preload)

	model = Sequential()

	model.add(Convolution2D(nb_filter=32,nb_row=3,nb_col=3,
							input_shape=(1, 128, 128),
							W_constraint=maxnorm(3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D())
	model.add(Dropout(0.2))

	model.add(Convolution2D(32,2,2,W_constraint=maxnorm(3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D())
	model.add(Dropout(0.4))

	model.add(Convolution2D(64,2,2,W_constraint=maxnorm(3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D())
	model.add(Dropout(0.5))

	model.add(Convolution2D(64,2,2,W_constraint=maxnorm(3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D())
	model.add(Dropout(0.5))

	model.add(Convolution2D(128,2,2,W_constraint=maxnorm(3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D())
	model.add(Dropout(0.5))

	model.add(Flatten())
	model.add(Dense(1024,W_constraint=maxnorm(3)))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	model.add(Dense(512,W_constraint=maxnorm(3)))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	model.add(Dense(1, activation='sigmoid'))

	model.compile(optimizer='adam', loss='binary_crossentropy')

	print 'Model compiled.'
	return model

def DataGen():
	with h5py.File('data') as hf:
		cats, dogs = hf.get('data')

	global validation_data, samples_per_epoch

	cat_train, cat_val = np.split(cats,[0.8*len(cats)])
	dog_train, dog_val = np.split(dogs, [0.8*len(dogs)])

	val_x = np.concatenate((cat_val, dog_val))
	val_y = np.concatenate((np.zeros(len(cat_val),dtype=np.float32),
							np.ones(len(dog_val),dtype=np.float32)))
	validation_data = (val_x, val_y)

	mini_batch_sz = 32
	samples_per_epoch = 6400

	y = np.concatenate((np.zeros(mini_batch_sz>>1,dtype=np.float32),
							np.ones(mini_batch_sz>>1,dtype=np.float32)))
	while 1:
		indices = np.random.randint(len(cat_train),size=mini_batch_sz>>1)
		eightCats, eightDogs = cat_train[indices], dog_train[indices]
		X = np.concatenate((eightCats, eightDogs))

		yield X,y

def runner(model, epochs):
	global samples_per_epoch, cat_val, dog_val
	training_data = DataGen()
	training_data.next()
	try:
		model.fit_generator(training_data,samples_per_epoch,epochs,
						verbose=1,validation_data=validation_data)
	except Exception as e:
		print e
	finally:
		fname = dumper(model,'cnn')
		print 'Model saved to disk at {}'.format(fname)
		return model

def main():
	model = init_model()
	return runner(model, 1000)

if __name__ == '__main__':
	main()