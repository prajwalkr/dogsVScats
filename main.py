import argparse
import sys

from scripts.utils import *
import scripts.cnn as CNN

def parse_args(args):
	parser = argparse.ArgumentParser(description='Kaggle Cats vs Dogs Redux Edition')
	parser.add_argument('-op',action='store',
						dest='op',choices=['test','train','vis','kaggle'],
						help='[test|train|vis|kaggle] operation',required=True)

	parser.add_argument('-cnn',action='store',default=False)
	parser.add_argument('-inp',action='store',default=None)

	args = parser.parse_args(args)
	operation = args.op
	cnn = args.cnn

	if cnn is False:
		raise ValueError('Choose a model to test on!')

	if type(cnn) == str and cnn.lower() == 'none':
		cnn = None

	if operation in ['train','vis']:
		return (operation,cnn,None)
	return (operation,cnn,args.inp)

def main(args):
	print 'Parsing arguments...'
	operation, cnn, inp_img = parse_args(args)

	if operation == 'test':
		if cnn:
			cnnModel = CNN.init_model(preload=cnn)
		else:
			cnnModel = CNN.init_model(compileModel=True)
		print 'Loaded CNN model'
		print 'Testing on input image(s)...'
		print tester(cnnModel,inp_img)

	elif operation == 'train':
		if cnn is not False:
			if type(cnn) == str:
				cnn = CNN.init_model(preload=cnn)
			else:
				cnn = CNN.init_model()

			cnnModel = CNN.main(cnn)
			return cnnModel

	elif operation == 'vis':
		if cnn is None:
			model = CNN.init_model(compileModel=True)
		if type(cnn) == str:
			model = CNN.init_model(preload=cnn)
		visualizer(model)
	
	elif operation == 'kaggle':
		if cnn:
			cnnModel = CNN.init_model(preload=None,compileModel=True)
			print 'Loaded CNN model'
			print 'Testing on input images...'
			CNN.kaggleTest(cnnModel)
			
if __name__ == '__main__':
	main(sys.argv[1:])