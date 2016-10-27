from utils import resizer, tester
from os import dirname, listdir
from os.path import abspath
from keras.models import load_model

MODELS_DIR = dirname(dirname(abspath(__file__))) + '/models/'

def get_model_list(prefix='vggfinetune'):
	return [MODELS_DIR + fname for fname in listdir(MODELS_DIR)
			if fname.startswith(prefix)]

def init_model(path):
	return load_model(path)
def main():
	for fname in get_model_list():
		model = load_model(fname)
		