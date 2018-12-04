import os
import tensorflow as tf
import numpy as np

STRAIGHT   = 0
LEFT       = 1
RIGHT      = 2
ACCELERATE = 3
BRAKE      = 4

def one_hot(labels):
    """
    this creates a one hot encoding from a flat vector:
    i.e. given y = [0,2,1]
     it creates y_one_hot = [[1,0,0], [0,0,1], [0,1,0]]
    """
    classes = np.unique(labels)
    n_classes = classes.size
    one_hot_labels = np.zeros(labels.shape + (n_classes,))
    for c in classes:
        one_hot_labels[labels == c, c] = 1.0
    return one_hot_labels

def rgb2gray(rgb):
    """
    this method converts rgb images to grayscale.
    """
    gray = np.dot(rgb[...,:3], [0.2125, 0.7154, 0.0721])
    return gray.astype('float32')

def encode_targets(y):
    """
    encoding scheme is one hot for commands: [LEFT, RIGHT, ACCEL, BRAKE, NOTHING]
    in my recording, breaking showed up as action [0.0, 0.0, 0.2], so some special handling in action_to_id
    """
    return one_hot(np.array(list(map(action_to_id, y))))


def action_to_id(a):
    """
    this method discretizes actions
    """

    # note: multiple actions found in the recording (example: steer left and brake)
    # here steering takes precedence over braking
    # and acceleration take precedence over braking :-)

    if a[0] == -1:
        return LEFT
    elif a[0] == 1:
        return RIGHT
    elif a[1] == 1:
        return ACCELERATE
    elif abs(a[2] - 0.2) < 0.01:
        return BRAKE
    else:
        return STRAIGHT

def sample_batch(X, y, batch_size, history_len, fair = False):

    # this sampling method samples uniformly over the actions
    # but it did not improve behavior in my case, so it is not used anymore
    if fair:
        indexes = []
        for i in range(5):
            case_i = np.where(y[:,i] == 1)[0]
            indexes.append(np.random.choice(case_i, batch_size))

        indexes = np.concatenate(indexes)
        indexes = indexes[indexes>(history_len-1)]
        indexes = np.random.choice(indexes, batch_size)
    else:
        # in non-fair sampling, simply choose a batch randomly from training data
        indexes = np.random.choice(np.arange((history_len-1), len(X)), batch_size)

    # in non-fair sampling, simply choose a batch randomly from training data
    #indexes = np.random.choice(np.arange((history_len-1), len(X)), batch_size)

    X_batch = np.zeros((batch_size, 96, 96, history_len))
    y_batch = np.zeros((batch_size, 5))

    for i, sel in enumerate(indexes):
        for j in range(history_len):
            X_batch[i,:,:,j] = X[sel-history_len+1+j, :, :]
        y_batch[i] = y[sel]

    return X_batch, y_batch

def clear_tensorboard(modelname):
    # kill old tensorboard values on restart
    tensorboard_path = os.path.join('./tensorboard', modelname)
    if tf.gfile.Exists(tensorboard_path):
        tf.gfile.DeleteRecursively(tensorboard_path)
