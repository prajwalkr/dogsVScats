from os import listdir
from os.path import abspath, dirname
import numpy as np
import cv2

ROOT = dirname(dirname(abspath(__file__)))
TRAIN_DIR = ROOT + '/train/'
channels, img_width, img_height = 3, 224, 224

def main():
	sum_img = np.zeros((channels, img_width, img_height),dtype=np.float32)
	CAT_PATH = TRAIN_DIR + 'cats/'
	for fname in listdir(CAT_PATH):
		sum_img += cv2.resize(cv2.imread(CAT_PATH + fname), (img_width, img_height)).transpose((2,0,1))
	DOG_PATH = TRAIN_DIR + 'dogs/'
	for fname in listdir(DOG_PATH):
		sum_img += cv2.resize(cv2.imread(DOG_PATH + fname), (img_width, img_height)).transpose((2,0,1))
	return sum_img / 25000.