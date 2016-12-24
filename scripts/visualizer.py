from keras import backend as K
from utils import read_image, write_image
import h5py
import numpy as np
from random import sample
from keras.models import load_model

img_width, img_height = 224, 224

def init_model(preload):
	return load_model(preload)

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def visualize(model, layer_name):
	print 'Model loaded.'
	layer_dict = dict([(layer.name, layer) for layer in model.layers])

	for filter_index in sample(range(0, layer_dict[layer_name].nb_filter),10):
		layer_output = layer_dict[layer_name].output
		loss = K.mean(layer_output[:, filter_index, :, :])
		grads = K.gradients(loss, model.layers[0].input)[0]
		grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
		iterate = K.function([model.layers[0].input, K.learning_phase()], [loss, grads])

		input_img_data = np.asarray([read_image('visimage.jpg')])

		for _ in xrange(100):
			loss_value, grads_value = iterate([input_img_data, 0])
			input_img_data += grads_value * 3

		img = deprocess_image(input_img_data[0])
		write_image(img, '../activations/out{}.jpg'.format(filter_index))

if __name__ == '__main__':
	visualize(init_model('bestyet.h5'), 'convolution2d_10')