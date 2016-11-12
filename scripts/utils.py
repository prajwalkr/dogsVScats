from keras.utils.visualize_util import plot
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import h5py, pickle
from os.path import abspath, dirname
from os import listdir
import scipy as sp
from datetime import datetime
from builder import prep_data
from shutil import copyfile

ROOT = dirname(dirname(abspath(__file__)))
TEST_DIR = ROOT + '/test/'
channels, img_width, img_height = 3, 150, 150
mini_batch_sz = 4
ext = '.jpg'

def logloss(act, pred):
    epsilon = 1e-15
    pred = max(epsilon, pred)
    pred = min(1-epsilon, pred)
    ll = act*sp.log(pred) + (1-act)*sp.log(1-pred)
    return ll

def visualizer(model):
	plot(model, to_file=ROOT + '/vis.png', show_shapes=True)

def test_data_gen(fnames):
	return prep_data(fnames[i:min(i + batch_size,len(fnames))])

def kaggleTest(model):
	fnames = [TEST_DIR + fname for fname in listdir(TEST_DIR)]

	ids = [x[:-4] for x in [fname for fname in listdir(TEST_DIR)]]
	X = prep_data(fnames)
	y = model.predict(X,batch_size=mini_batch_sz,verbose=1)

	with open(ROOT + 'out.csv','w') as f:
		f.write('id,label\n')
		for i,pred in zip(ids,y):
			f.write('{},{}\n'.format(i,str(pred[1])))
	return zip(ids, y)
'''
def tester(topModel,vgg=None,img_path=None):
	if img_path is None:
		test_gen = ImageDataGenerator()
		test_gen = test_gen.flow_from_directory(
			TEST_DIR,target_size=(img_width, img_height),
			batch_size=mini_batch_sz,
			class_mode=None)
		i = 0
		cl, dl = 0., 0.
		total_samples = len(listdir(TEST_DIR + 'cats')) + len(listdir(TEST_DIR + 'dogs'))
		while i < total_samples:
			X = test_gen.next()
			y = topModel.predict(X, batch_size=8)
			if i % 3 == 0: print i
			for pred in y[:len(X)/2]:
				cl += logloss(0, pred[0])
			for pred in y[len(X)/2:]:
				dl += logloss(1, pred[0])
			i += len(X)
		print cl
		print dl
		ll = cl + dl
		print -ll/total_samples
		return y
	img = cv2.imread(img_path,0 if CHANNELS == 1 else 3)
	img = cv2.resize(img, (ROW,COL))
	x = img.astype(np.float32)
	x = x.reshape(1,CHANNELS,ROW,COL)
	y = topModel.predict(x)[0]
	print y
'''
def dumper(model,kind,fname=None):
	if not fname:
		fname = '{}/models/{}-{}.h5'.format(ROOT,
										str(datetime.now()).replace(' ','-'),kind)
	try:
		with open(fname,'w') as f:
			model.save(fname)
	except IOError:
		raise IOError('Unable to open: {}'.format(fname))
	return fname

def segTest(model):
	pairs = kaggleTest(model)
	for iD, confidence in pairs:
		if confidence[0] > 0.999:
			copyfile(TEST_DIR + iD + ext, ROOT + '/testing/dogs/' + iD + ext)
		if confidence[0] < 0.001:
			copyfile(TEST_DIR + iD + ext, ROOT + '/testing/cats/' + iD + ext)
