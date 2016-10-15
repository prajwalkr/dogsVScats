import os
import cv2
import numpy as np
import h5py

TRAIN_DIR = 'train/'
IMG_SHAPE = (128,128)
CHANNELS = 1 # grayscale

catImages = [TRAIN_DIR + fname for fname in os.listdir(TRAIN_DIR) if 'cat' in fname]
dogImages = [TRAIN_DIR + fname for fname in os.listdir(TRAIN_DIR) if 'dog' in fname]

def read_image(file_path):
    img = cv2.imread(file_path, 0)
    return cv2.resize(img, IMG_SHAPE, interpolation=cv2.INTER_CUBIC)

def prep_data(images):
    count = len(images)
    data = np.ndarray((count, CHANNELS, IMG_SHAPE[0], IMG_SHAPE[1]), dtype=np.float32)

    for i, image_file in enumerate(images):
        image = read_image(image_file)
        data[i] = image.T
        if i%1000 == 0: print('Processed {} of {}'.format(i, count))
    
    data /= 255.
    return data

def dumper(data,name='data'):
	with h5py.File(name) as hf:
		hf.create_dataset(name,data=data)

cats = prep_data(catImages)
dogs = prep_data(dogImages)

dumper(np.asarray([cats, dogs]))