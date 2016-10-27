from keras.utils.visualize_util import plot
from keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import h5py, pickle
from os.path import abspath, dirname
from os import listdir
import scipy as sp
from datetime import datetime
from builder import prep_data

ROOT = dirname(dirname(abspath(__file__)))

def logloss(act, pred):
    epsilon = 1e-15
    pred = max(epsilon, pred)
    pred = min(1-epsilon, pred)
    ll = act*sp.log(pred) + (1-act)*sp.log(1-pred)
    return ll

def visualizer(model):
	plot(model, to_file=ROOT + 'vis.png', show_shapes=True)

def resizer(X,shape):
	if len(shape) != 4 or shape[0] != len(X): raise ValueError("Shape improper")
	l, CHANNELS, ROW, COL = shape
	Y = np.ndarray((l, CHANNELS, ROW, COL), dtype=np.float32)
	for i in xrange(len(X)):
		Y[i] = cv2.resize(X[i].T,(ROW,COL)).T
	return Y

def test_data_gen(fnames):
	return prep_data(fnames[i:min(i + batch_size,len(fnames))])

def kaggleTest(model):
	TEST_DIR = ROOT + '/test/'
	fnames = [TEST_DIR + fname for fname in listdir(TEST_DIR)]

	ids = [x[:-4] for x in [fname for fname in listdir(TEST_DIR)]]
	X = prep_data(fnames)
	y = model.predict(X,val_samples=len(fnames))

	with open(ROOT + 'out.csv','w') as f:
		f.write('id,label\n')
		for i,pred in zip(ids,y):
			f.write('{},{}\n'.format(i,str(pred[0])))

def tester(topModel,vgg=None,img_path=None):
	if img_path is None:
		path = ROOT + '/test.h5'
		with h5py.File(path) as hf:
			cats, dogs = hf.get('data')
		X = np.concatenate((cats, dogs))
		if vgg:
			X = resizer(X, (len(X),) + vgg.layers[0].input_shape[1:])
			X = vgg.predict(X, verbose=1)
		else:
			X = resizer(X, (len(X),) + topModel.layers[0].input_shape[1:])
		
		y = topModel.predict(X,batch_size=8,verbose=1)
		ll = 0.0
		for pred in y[:len(X)/2]:
			ll += logloss(0, pred[0])
		for pred in y[len(X)/2:]:
			ll += logloss(1, pred[0])
		print -ll/len(y)
		return y
	img = cv2.imread(img_path,0 if CHANNELS == 1 else 3)
	img = cv2.resize(img, (ROW,COL))
	x = img.astype(np.float32)
	x = x.reshape(1,CHANNELS,ROW,COL)
	y = topModel.predict(x)[0]
	print y

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