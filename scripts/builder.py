import os
import cv2
import numpy as np
import h5py
import sys

### Defining constants...
TRAIN_DIR = '../train/'
TEST_DIR = '../test/'
IMG_SHAPE = (64,64)
CHANNELS = 3


def read_image(file_path):
    img = cv2.imread(file_path)
    return cv2.resize(img, IMG_SHAPE, interpolation=cv2.INTER_CUBIC)

def prep_data(images):
    count = len(images)
    data = np.ndarray((count, CHANNELS, IMG_SHAPE[0], IMG_SHAPE[1]), dtype=np.float32)

    for i, image_file in enumerate(images):
        image = read_image(image_file)
        data[i] = image.T
        if i%1000 == 0: print('Processed {} of {}'.format(i, count))
    
    return data

def dumper(data,name='data'):
	with h5py.File('../' + name + '.h5') as hf:
		hf.create_dataset(name,data=data)

def main(op):
    if op not in ['train','test']:
        raise ValueError('Operation {} not supported!'.format(op))

    if op == 'train':
        print 'Building training data..'
        catImages = [TRAIN_DIR + fname for fname in os.listdir(TRAIN_DIR) if 'cat' in fname]
        dogImages = [TRAIN_DIR + fname for fname in os.listdir(TRAIN_DIR) if 'dog' in fname]
        splitcat, splitdog = len(catImages) >> 2, len(dogImages) >> 1
        cat_train, cat_test = catImages[:-splitcat],catImages[-splitcat:]
        dog_train, dog_test = dogImages[:-splitdog],dogImages[-splitdog:]
        cats = prep_data(cat_train) 
        dogs = prep_data(dog_train) 
        dumper(np.asarray([cats,dogs]))
        cats = prep_data(cat_test) 
        dogs = prep_data(dog_test)
        dumper(np.asarray([cats,dogs]))
    else:
        print 'Building testing data..'
        images = np.asarray([prep_data([TEST_DIR + fname for fname in os.listdir(TEST_DIR)])])
        dumper(images)

if __name__ == '__main__':
    main(sys.argv[1])
