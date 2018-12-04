import pickle
import gzip
from model1 import Model1
from utils import clear_tensorboard

X_train, y_train, X_valid, y_valid = pickle.load(gzip.open('./dataset_fast.cache', 'rb'))

clear_tensorboard('model1')

model = Model1(lr=0.0001, name="model1")
model.train(X_train, y_train, X_valid, y_valid, n_minibatches=100000, snapshos=[30000])
model.sess.close()

