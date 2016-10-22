from keras.utils.visualize_util import plot
import cv2
import numpy as np
import h5py
from os.path import abspath, dirname
import scipy as sp
from datetime import datetime

ROW, COL, CHANNELS = 227,227,3
ROOT = dirname(dirname(abspath(__file__)))

def logloss(act, pred):
    epsilon = 1e-15
    pred = max(epsilon, pred)
    pred = min(1-epsilon, pred)
    ll = act*sp.log(pred) + (1-act)*sp.log(1-pred)
    return ll

def visualizer(model):
	plot(model, to_file='vis.png', show_shapes=True)

def resizer(X):
	Y = np.ndarray((len(X), CHANNELS, ROW, COL), dtype=np.float32)
	for i in xrange(len(X)):
		Y[i] = cv2.resize(X[i].T,(ROW,COL)).T
	return Y

def tester(model,img_path=None):
	if img_path is None:
		path = ROOT + '/test.h5'
		with h5py.File(path) as hf:
			cats, dogs = hf.get('data')
			X = 255*resizer(np.concatenate((cats, dogs)))
			y = model.predict(X,verbose=1)
			ll = 0.0
			for pred in y[:len(X)/2]:
				ll += logloss(0, pred[0])
			for pred in y[len(X)/2:]:
				ll += logloss(1, pred[0])
			print -ll/len(X)
		return 
	img = cv2.imread(img_path,0 if CHANNELS == 1 else 3)
	img = cv2.resize(img, (ROW,COL))
	x = img.astype(np.float32)
	x = x.reshape(1,CHANNELS,ROW,COL)
	y = model.predict(x)[0]
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