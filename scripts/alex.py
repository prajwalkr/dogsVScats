from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, \
	Input, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.constraints import maxnorm
import numpy as np
from customlayers import convolution2Dgroup, crosschannelnormalization, \
	splittensor, Softmax4D
from os.path import dirname, abspath
from keras.regularizers import l2

def pop_layer(model):
	model.layers[-1].outbound_nodes = []
	model.outputs = [model.layers[-1].output]
	model.built = False
	return model

def alex(weights_path=None):
	if not weights_path:
		weights_path = dirname(abspath(__file__)) + '/alexnet_weights.h5'

	inputs = Input(shape=(3,227,227))

	conv_1 = Convolution2D(96, 11, 11,subsample=(4,4),activation='relu',
						   name='conv_1',trainable=False)(inputs)

	conv_2 = MaxPooling2D((3, 3), strides=(2,2))(conv_1)
	conv_2 = crosschannelnormalization(name="convpool_1")(conv_2)
	conv_2 = ZeroPadding2D((2,2))(conv_2)
	conv_2 = merge([
		Convolution2D(128,5,5,activation="relu",name='conv_2_'+str(i+1),trainable=False)(
			splittensor(ratio_split=2,id_split=i)(conv_2)
		) for i in range(2)], mode='concat',concat_axis=1,name="conv_2")

	conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
	conv_3 = crosschannelnormalization()(conv_3)
	conv_3 = ZeroPadding2D((1,1))(conv_3)
	conv_3 = Convolution2D(384,3,3,activation='relu',name='conv_3',trainable=False)(conv_3)

	conv_4 = ZeroPadding2D((1,1))(conv_3)
	conv_4 = merge([
		Convolution2D(192,3,3,activation="relu",name='conv_4_'+str(i+1))(
			splittensor(ratio_split=2,id_split=i)(conv_4)
		) for i in range(2)], mode='concat',concat_axis=1,name="conv_4")

	conv_5 = ZeroPadding2D((1,1))(conv_4)
	conv_5 = merge([
		Convolution2D(128,3,3,activation="relu",name='conv_5_'+str(i+1))(
			splittensor(ratio_split=2,id_split=i)(conv_5)
		) for i in range(2)], mode='concat',concat_axis=1,name="conv_5")

	pool = MaxPooling2D((3, 3), strides=(2,2),name="convpool_5")(conv_5)

	flat = Flatten(name="flatten")(pool)
	dense_1 = Dense(4096, activation='relu',name='dense_1')(flat)
	dense_2 = Dropout(0.5)(dense_1)
	dense_2 = Dense(4096, activation='relu',name='dense_2')(dense_2)
	dense_3 = Dropout(0.5)(dense_2)
	dense_3 = Dense(1000,name='dense_3')(dense_3)
	prediction = Activation("softmax",name="softmax")(dense_3)

	model = Model(input=inputs, output=prediction)

	if weights_path:
		model.load_weights(weights_path)

	# finetune only the dense layers
	dense_1 = Dense(1024, activation='relu',name='dense_1',W_constraint=maxnorm(3),
					W_regularizer=l2())(flat)
	dense_2 = Dropout(0.5)(dense_1)
	dense_2 = Dense(512, activation='relu',name='dense_2',W_constraint=maxnorm(3),
					W_regularizer=l2())(dense_2)
	dense_3 = Dropout(0.5)(dense_2)
	dense_3 = Dense(1,name='dense_3',activation='sigmoid')(dense_3)

	model = Model(input=inputs, output=dense_3)

	return model