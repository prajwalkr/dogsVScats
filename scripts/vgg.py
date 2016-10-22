from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.constraints import maxnorm
from os.path import dirname, abspath

ROOT = dirname(dirname(abspath(__file__)))

def pop_layer(model):
    model.layers[-1].outbound_nodes = []
    model.outputs = [model.layers[-1].output]
    model.built = False
    return model
    
def VGG_16(weights_path=None):
    model = Sequential()
    inputs = Input(shape=(3,224,224))
    zero_pad_1 = ZeroPadding2D((1,1),input_shape=(3,224,224))(inputs)
    conv_1 = Convolution2D(64, 3, 3, activation='relu',trainable=False)(zero_pad_1)
    zero_pad_2 = ZeroPadding2D((1,1))(conv_1)
    conv_2 = Convolution2D(64, 3, 3, activation='relu',trainable=False)(zero_pad_2)
    pool_1 = MaxPooling2D((2,2), strides=(2,2))(conv_2)

    zero_pad_3 = ZeroPadding2D((1,1))(pool_1)
    conv_3 = Convolution2D(128, 3, 3, activation='relu',trainable=False)(zero_pad_3)
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu',trainable=False))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu',trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu',trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu',trainable=False))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu',trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu',trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu',trainable=False))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    weights_path = ROOT + '/vgg16_weights.h5'
    if weights_path:
        model.load_weights(weights_path)

    for _ in xrange(6):
        model = pop_layer(model)

    model.add(Flatten())
    model.add(Dense(1024, activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    return model
