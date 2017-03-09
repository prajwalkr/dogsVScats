from resnet import ResNet50
from keras.models import Sequential, load_model, Model
from keras.layers import merge
from os.path import dirname, abspath
from os import listdir
import cnn as c
from utils import kaggleTest
import pickle

img_side = 300
ROOT = dirname(dirname(abspath(__file__)))
head_start = 174

def init_ensemble(paths):
	weights_path = ROOT + '/resnet50_tf_notop.h5'
	ensemble = Sequential()
	ensemble.add(ResNet50(input_shape=(img_side, img_side, 3), weights_path=weights_path))
	heads = [ensemble.output for _ in xrange(len(paths))]
	for i, path in enumerate(paths):
		temp_model = load_model(path)
		for layer in temp_model.layers[head_start:]:
			print layer.name
			heads[i] = layer (heads[i])
	# outs = [head (ensemble) for head in heads]
	merged = merge(outs, mode='ave')
	return Model(ensemble.input, merged)

def kaggle_ensemble(models):
	outs = []
	for modelpath in models:
		if modelpath in listdir('../kaggle/'): outs.append(pickle.load(open('../kaggle/' + modelpath)))
		else:
			model = c.init_model(modelpath, declare=False)
			out = kaggleTest(model, write_csv=False, img_side = model.input_shape[1], 
				inception='inception' in modelpath)
			pickle.dump(out, open('../kaggle/' + modelpath, 'w'))
			outs.append(out)
	if len(outs) == 1: outs.append(outs[0])
	dog_probabs = [sum([m,n,o]) / len(outs) for m,n,o in zip(*outs)]
	kaggleTest(None, predict=False, dog_probabs=dog_probabs)

if __name__ == '__main__':
	include = [
	'bestvalyet.h5',
	'bestvalinception11.h5',
	]
	kaggle_ensemble(include)