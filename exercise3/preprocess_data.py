# helper script - preprocesses data
# and creates a file "dataset_fast.cache"
# for fast data loading

import pickle
import numpy as np
import os
import gzip
from utils import *

def load(file_pattern, frm, to, frac = 0.1):
    """Customized load method for my data - the data is split up into multiple numbered recordings, this
       method joins them together, splits them into train and validation and returns them"""
    states = []
    actions = []

    print("loading...")

    for i in range(frm,to+1):
        # insert number into file name template
        data_file = file_pattern.replace("$i$", str(i))
        print("  loading data from", data_file)

        f = gzip.open(data_file,'r')
        data = pickle.load(f)
        states.append(np.stack( data['state'] ))
        actions.append(np.stack( data['action']))

    # join np arrays from files together
    X = np.vstack(states)
    y = np.vstack(actions)

    # slipt into train and validation
    n_samples = len(X)
    X_train, y_train = X[:int((1-frac) * n_samples)], y[:int((1-frac) * n_samples)]
    X_valid, y_valid = X[int((1-frac) * n_samples):], y[int((1-frac) * n_samples):]

    # some state
    print(X_train.shape[0], "training samples", X_valid.shape[0], "validation samples", )

    return X_train, y_train, X_valid, y_valid


def preprocessing(X, y):

    # History:
    # At first you should only use the current image as input to your network to learn the next action. Then the input states
    # have shape (96, 96,1). Later, add a history of the last N images to your state so that a state has shape (96, 96, N).

    print("preprocessing...")
    print("  transforming to greyscale")
    X = rgb2gray(X)

    print("  encoding targets")
    y = encode_targets(y)

    print("  preprocessing done")
    return X, y


# cache preprocessed data for faster development
cachefile = "dataset_fast.cache"

if not os.path.exists(cachefile):
    # read data from cached file

    # read datasets 1-9

    # using all 9 datasets here
    X_train, y_train, X_valid, y_valid = load("./data_raw/datanew$i$.pkl.gzip", 1, 9)

    # preprocess data
    X_train, y_train = preprocessing(X_train, y_train)
    X_valid, y_valid = preprocessing(X_valid, y_valid)

    # output some info
    print(X_train.shape)
    print(y_train.shape)
    print(X_valid.shape)
    print(y_valid.shape)

    pickle.dump([X_train, y_train, X_valid, y_valid], gzip.open(cachefile, 'wb'))
else:
    print("Data file exists - delete to reprocess")