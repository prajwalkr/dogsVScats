from os import listdir
from os.path import abspath, dirname
import numpy as np
import cv2
from utils import read_image
from keras.preprocessing.image import ImageDataGenerator
import pickle

ROOT = dirname(dirname(abspath(__file__)))
TRAIN_DIR = ROOT + '/train/'
channels, img_width, img_height = 3, 224, 224
num_cats_train = len(listdir(TRAIN_DIR + '/cats'))
num_dogs_train = len(listdir(TRAIN_DIR + '/dogs'))
num_images = num_cats_train + num_dogs_train
num_iterations = 5
batch_sz = 10

def main():
	dg = ImageDataGenerator(horizontal_flip=True, rotation_range=20., width_shift_range=0.2, 
		height_shift_range=0.2)
	dg = dg.flow_from_directory(
		TRAIN_DIR,
		target_size=(img_width, img_height),
		batch_size=batch_sz,
		class_mode='categorical')

	mean = np.zeros((channels, img_width, img_height),dtype=np.float32)
	e_x2byN = np.zeros((channels, img_width, img_height),dtype=np.float32)

	N = num_images * num_iterations
	for count in xrange(num_iterations):
		print 'Iteration {}/{}'.format(count + 1, num_iterations)
		cur_count = 0
		while cur_count < num_images:
			x,y = dg.next()
			for img in x:
				mean += (img.astype(np.float32) / N)
				e_x2byN += ((img.astype(np.float32) ** 2) / N)
			cur_count += batch_sz

	stddev = (e_x2byN - mean ** 2) ** 0.5
	pickle.dump((mean, stddev), open('meanSTDDEV','w'))
	return mean, stddev

if __name__ == '__main__':
	main()