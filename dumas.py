from os import listdir, remove

VAL_PATH = "validation/cats/"
TRAIN_PATH = 'train/cats/'

d = dict()

for fname in listdir(VAL_PATH):
	d[fname]=True

for fname in listdir(TRAIN_PATH):
	if fname in d:
		remove(TRAIN_PATH + fname)

