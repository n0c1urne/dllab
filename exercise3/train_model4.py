import pickle
import gzip
from model4 import Model4
from utils import clear_tensorboard

X_train, y_train, X_valid, y_valid = pickle.load(gzip.open('./dataset_fast.cache', 'rb'))

clear_tensorboard('model4')

model = Model4(lr=0.0001, name="model4", history_len=10)

# train for 50000 batches
model.train(X_train, y_train, X_valid, y_valid, n_minibatches=100000, snapshots=[10000, 20000, 30000, 40000])
model.sess.close()

