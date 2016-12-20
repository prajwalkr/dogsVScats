import cnn as c
from utils import dog_probab as dp
from utils import logloss

main = c.init_model('bestyet.h5')
ft = c.init_model('current.h5')

tg, vg = c.DataGen()
x = 0
preds, actual = [], []

def doubtful(x):
	return x > 0.4 and x < 0.6

while x < 5716:
	X,y = vg.next()
	Y = dp(main.predict(X))
	actual += dp(y)
	for j, pred in enumerate(Y):
		if doubtful(pred):
			fty = dp(ft.predict(X[j].reshape(1,3,224,224)))[0]
			Y[j] = min(fty, Y[j]) if Y[j] < 0.5 else max(fty, Y[j])
	preds += Y
	x += 8

print logloss(actual, preds)


