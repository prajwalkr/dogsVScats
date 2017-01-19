from keras.optimizers import SGD
from keras.callbacks import Callback, ModelCheckpoint, CSVLogger, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pickle
from utils import resize
from os import listdir
from os.path import dirname, abspath
from random import sample, randint
from utils import *

ROOT = dirname(dirname(abspath(__file__)))
TRAIN_DIR, VAL_DIR = ROOT + '/train', ROOT + '/validation'
num_cats_train = len(listdir(TRAIN_DIR + '/cats'))
num_dogs_train = len(listdir(TRAIN_DIR + '/dogs'))
num_cats_val = len(listdir(VAL_DIR + '/cats'))
num_dogs_val = len(listdir(VAL_DIR + '/dogs'))
samples_per_epoch = num_cats_train + num_dogs_train
nb_val_samples = num_cats_val + num_dogs_val

channels, img_side = 3, 224
batch_sz = 16
mean, stddev = pickle.load(open('meanSTDDEV'))

def logloss(preds, truths):
	epsilon = 1e-15
	preds = np.clip(preds, epsilon, 1 - epsilon)
	return [(truth * np.log(pred) + (1 - truth) * np.log(1 - pred))[0] for truth, pred in zip(truths, preds)]

def standardized(X):
	for i in xrange(len(X)):
		if randint(0,5)//5:
			X[i] = blur(X[i])
		if randint(0, 5)//5:
			X[i] = random_bright_shift(X[i])
		if randint(0, 5)//5:
			X[i] = random_contrast_shift(X[i])
		X[i] = (X[i] - mean) / stddev
	return X

def standardized_gen(gen):
	while 1:
		X,y = gen.next()
		yield standardized(X), y

class SpacedRep(Callback):
	def __init__(self):
		self.num_boxes = 3
		self.boxes = [dict() for _ in xrange(self.num_boxes)]
		self.imgName2box = dict()
		print 'Adding images to Box 0...'
		for fname in listdir(TRAIN_DIR + '/cats'):
			full_fname = '{}/cats/{}'.format(TRAIN_DIR, fname)
			self.boxes[0][full_fname] = True
			self.imgName2box[full_fname] = 0
		for fname in listdir(TRAIN_DIR + '/dogs'):
			full_fname = '{}/dogs/{}'.format(TRAIN_DIR, fname)
			self.boxes[0][full_fname] = True
			self.imgName2box[full_fname] = 0
		# self.default_nums = [int(0.5 * batch_sz), int(0.375 * batch_sz),
		# 			 int(0.125 * batch_sz)] # 0.5 of whole, 0.375, 0.125
		self.default_nums = [13,2,1]
		self.cur_batch = []
		self.cur_XY = None
		self.remaining = samples_per_epoch

	def get_next_batch(self):
		while 1:
			nums = self.get_nums_for_boxes()
			X = np.ndarray((min(self.remaining, batch_sz), channels, img_side, img_side), dtype=np.float32)
			Y = np.ndarray((min(self.remaining, batch_sz)), dtype=np.float32)
			self.cur_batch = []
			cur = 0
			for i in xrange(self.num_boxes):
				for fname in sample(self.boxes[i], int(nums[i])):
					self.cur_batch.append(fname)
					X[cur] = read_image(fname)
					if 'cats' in fname: Y[cur] = 0
					else: Y[cur] = 1
					cur += 1
			# print len(self.boxes[0]), len(self.boxes[1]), len(self.boxes[2])
			self.remaining -= len(X)
			X = standardized(X)
			self.cur_XY = (X, Y)
			yield X, Y
			self.move_to_new_box()
			if self.remaining <= 0:
				self.remaining = samples_per_epoch
				print len(self.boxes[0]), len(self.boxes[1]), len(self.boxes[2])

	# Utils
	def get_nums_for_boxes(self):
		nums = self.default_nums
		if nums[0] > len(self.boxes[0]):
			nums[1] += nums[0] - len(self.boxes[0])
			nums[0] = len(self.boxes[0])
		if nums[1] > len(self.boxes[1]):
			nums[2] += nums[1] - len(self.boxes[1])
			nums[1] = len(self.boxes[1])
		if nums[2] > len(self.boxes[2]):
			nums[0] += nums[2] - len(self.boxes[2])
			nums[2] = len(self.boxes[2])
		assert sum(nums) == min(self.remaining, batch_sz)
		for i in xrange(self.num_boxes):
			assert nums[i] <= len(self.boxes[i])
		return nums

	def move_to_new_box(self):
		preds = model.predict(self.cur_XY[0]) # list of losses
		# mean_loss = sum(losses) / len(losses)
		# sorted_losses = sorted([(loss, i) for i, loss in enumerate(losses)])
		# box_allocs = (sorted_losses[:self.default_nums[0]],
		# 				sorted_losses[self.default_nums[0]:sum(self.default_nums[:2])],
		# 				sorted_losses[sum(self.default_nums[:2]):sum(self.default_nums)])
		def doubtful(pred):
			return pred > 0.2 and pred < 0.8

		for i, loss in enumerate(preds):
			fname = self.cur_batch[i]
			cur_box_no = self.imgName2box[fname]
			if doubtful(preds[i][0]):
				self.imgName2box[fname] = 0
				del self.boxes[cur_box_no][fname]
				self.boxes[0][fname] = True
			else:
				self.imgName2box[fname] = min(self.num_boxes - 1, cur_box_no + 1)
				del self.boxes[cur_box_no][fname]
				self.boxes[min(self.num_boxes - 1, cur_box_no + 1)][fname] = True

		# for new_box_no, alloc in enumerate(box_allocs):
		# 	for _, i in alloc:
		# 		fname = self.cur_batch[i]
		# 		cur_box_no = self.imgName2box[fname]
		# 		self.imgName2box[fname] = new_box_no
		# 		del self.boxes[cur_box_no][fname]
		# 		self.boxes[new_box_no][fname] = True

		
def spacedRunner(m, epochs=100):
	S = SpacedRep()
	global model
	model = m
	train_gen = S.get_next_batch()
	val_datagen = standardized_gen(ImageDataGenerator(horizontal_flip=True).flow_from_directory(
		VAL_DIR,target_size=(img_side, img_side),
		batch_size=batch_sz,
		class_mode='binary'))

	model.compile(optimizer=SGD(1e-3, momentum=0.9, nesterov=True), loss='binary_crossentropy')

	val_checkpoint = ModelCheckpoint('bestval.h5','val_loss',1,True)
	cur_checkpoint = ModelCheckpoint('current.h5')

	print 'Model compiled.'

	try:
		model.fit_generator(train_gen,samples_per_epoch,epochs,
						verbose=1,validation_data=val_datagen,nb_val_samples=nb_val_samples,
						callbacks=[val_checkpoint, cur_checkpoint])
	except Exception as e:
		print e
	finally:
		fname = dumper(model,'cnn')
		print 'Model saved to disk at {}'.format(fname)
		return model
