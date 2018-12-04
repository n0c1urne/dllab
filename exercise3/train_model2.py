import pickle
import gzip
from model2 import Model2
from utils import clear_tensorboard

X_train, y_train, X_valid, y_valid = pickle.load(gzip.open('./dataset_fast.cache', 'rb'))


clear_tensorboard('model2')
model = Model2(lr=0.0001, name='model2')

# train for 100000 batches for graph from report, snapshot at 30000 for best model, before overfitting
model.train(X_train, y_train, X_valid, y_valid, n_minibatches=100000, snapshots=[30000])
model.sess.close()

