# A simple lenet-like convolutional network to classify mnist digits in tensorflow

# run with the best parameters from random search:
#
# python3 cnn_mnist.py --epochs=20 --filter_size=5 --batch_size=35 --learning_rate=0.045 --num_filters=31

import argparse
import gzip
import os
import pickle
import numpy as np
from random import shuffle
import tensorflow as tf


def one_hot(labels):
    """this creates a one hot encoding from a flat vector:
    i.e. given y = [0,2,1]
     it creates y_one_hot = [[1,0,0], [0,0,1], [0,1,0]]
    """
    classes = np.unique(labels)
    n_classes = classes.size
    one_hot_labels = np.zeros(labels.shape + (n_classes,))
    for c in classes:
        one_hot_labels[labels == c, c] = 1
    return one_hot_labels


def mnist(datasets_dir='./data'):
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    data_file = os.path.join(datasets_dir, 'mnist.pkl.gz')
    if not os.path.exists(data_file):
        print('... downloading MNIST from the web')
        try:
            import urllib
            urllib.urlretrieve('http://google.com')
        except AttributeError:
            import urllib.request as urllib
        url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        urllib.urlretrieve(url, data_file)

    print('... loading data')
    # Load the dataset
    f = gzip.open(data_file, 'rb')
    try:
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
    except TypeError:
        train_set, valid_set, test_set = pickle.load(f)
    f.close()

    test_x, test_y = test_set
    test_x = test_x.astype('float32')
    test_x = test_x.astype('float32').reshape(test_x.shape[0], 28, 28, 1)
    test_y = test_y.astype('int32')
    valid_x, valid_y = valid_set
    valid_x = valid_x.astype('float32')
    valid_x = valid_x.astype('float32').reshape(valid_x.shape[0], 28, 28, 1)
    valid_y = valid_y.astype('int32')
    train_x, train_y = train_set
    train_x = train_x.astype('float32').reshape(train_x.shape[0], 28, 28, 1)
    train_y = train_y.astype('int32')
    print('... done loading data')
    return train_x, one_hot(train_y), valid_x, one_hot(valid_y), test_x, one_hot(test_y)


def shuffle_helper(x, y):
    ind_list = list(range(x.shape[0]))
    shuffle(ind_list)
    return x[ind_list, :, :, :], y[ind_list]


# a lenet-like network graph in tensorflow (not using keras :-( )
def network(x, num_filters, filter_size):
    mu = 0
    sigma = 0.1

    # in comments: example dimension calculation with filter_size = 3, num_filters = 16

    conv1_w = tf.Variable(tf.truncated_normal(shape=[filter_size, filter_size, 1, num_filters], mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(num_filters))
    conv1 = tf.nn.conv2d(x, conv1_w, strides=[1, 1, 1, 1], padding='SAME') + conv1_b
    conv1 = tf.nn.relu(conv1)
    # shape: (?, 28, 28, 16)

    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # shape: (?, 14, 14, 16)

    conv2_w = tf.Variable(tf.truncated_normal(shape=[filter_size, filter_size, num_filters, num_filters], mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(num_filters))
    conv2 = tf.nn.conv2d(pool1, conv2_w, strides=[1, 1, 1, 1], padding='SAME') + conv2_b
    conv2 = tf.nn.relu(conv2)
    # shape: (?, 14, 14, 16)

    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # shape: (?, 7, 7, 16)

    shape = pool2.get_shape().as_list()
    dim = np.prod(shape[1:])
    flat = tf.reshape(pool2, [-1, dim])
    # shape: (?, 784)

    fc1_w = tf.Variable(tf.truncated_normal(shape=(dim, 128), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(128))
    fc1 = tf.matmul(flat, fc1_w) + fc1_b
    # shape: (?, 128)

    fc2_w = tf.Variable(tf.truncated_normal(shape=(128, 10), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(10))
    fc2 = tf.matmul(fc1, fc2_w) + fc2_b
    # shape: (?, 10)

    return fc2

# a helper method, yields data in batches


def batched(x_data, y_data, batch_size):
    num_examples = len(x_data)
    for batch_index in range(0, num_examples, batch_size):
        last = batch_index + batch_size
        batch_x, batch_y = x_data[batch_index:last], y_data[batch_index:last]
        yield batch_x, batch_y


# evaluate dataset and get accuracy and loss
def evaluate(x_data, y_data, model, batch_size):
    x, y, accuracy_operation, loss_operation = model
    sess = tf.get_default_session()

    num_examples = len(x_data)

    total_accuracy = 0
    total_loss = 0

    # iterate batches
    for batch_x, batch_y in batched(x_data, y_data, batch_size):
        # run validation accuracy and determine loss on validation set
        accuracy, loss = sess.run([accuracy_operation, loss_operation], feed_dict={x: batch_x, y: batch_y})
        total_accuracy += accuracy * len(batch_x)
        total_loss += loss * len(batch_x)

    # validation and loss of this epoch
    validation_accuracy = total_accuracy / num_examples
    loss = total_loss / num_examples

    return validation_accuracy, loss


def train_and_validate(x_train, y_train, x_valid, y_valid, num_epochs, lr, num_filters, batch_size, filter_size):
    # create placeholders for input and labels, None for batch dimension
    x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    y = tf.placeholder(tf.int32, shape=(None))

    # create lenet like graph
    logits = network(x, num_filters, filter_size)

    # define loss, training, prediction and accuracy heads
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
    training_operation = optimizer.minimize(loss_operation)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # group placeholders, operations as model
    model = (x, y, accuracy_operation, loss_operation)

    # use default session for convenience
    sess = tf.get_default_session()
    sess.run(tf.global_variables_initializer())
    num_examples = len(x_train)

    # lists for storing history of training and validation accuracies/losses
    training_accuracies = []
    training_losses = []
    validation_accuracies = []
    validation_losses = []

    # train for some epochs
    for epoch in range(num_epochs):
        print("epoch "+str(epoch+1)+":")

        # shuffle data for batch run
        shuffled_x, shuffled_y = shuffle_helper(x_train, y_train)

        # iterate batches
        for batch_x, batch_y in batched(shuffled_x, shuffled_y, batch_size):
            # run training
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        # evaluate training and validation for accuracies and losses
        training_accuracy, training_loss = evaluate(x_train, y_train, model, batch_size)
        validation_accuracy, validation_loss = evaluate(x_valid, y_valid, model, batch_size)

        # collect accuracies and losses
        validation_accuracies.append(float(validation_accuracy))
        validation_losses.append(float(validation_loss))
        training_accuracies.append(float(training_accuracy))
        training_losses.append(float(training_loss))

        print("validation acc. = {:.4f}".format(validation_accuracy))
        print("validation loss = {:.4f}".format(validation_loss))

    # return accuracy and loss history, and input and output nodes as model
    return {
        "val_acc":  validation_accuracies,
        "val_loss": validation_losses,
        "train_acc":  training_accuracies,
        "train_loss": training_losses,
    }, model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default="./", type=str, nargs="?",
                        help="Path where the results will be stored")
    parser.add_argument("--input_path", default="./", type=str, nargs="?",
                        help="Path where the data is located. If the data is not available it will be downloaded first")
    parser.add_argument("--learning_rate", default=1e-3, type=float, nargs="?", help="Learning rate for SGD")
    parser.add_argument("--num_filters", default=16, type=int, nargs="?",
                        help="The number of filters for each convolution layer")
    parser.add_argument("--batch_size", default=32, type=int, nargs="?", help="Batch size for SGD")
    parser.add_argument("--epochs", default=10, type=int, nargs="?",
                        help="Determines how many epochs the network will be trained")
    parser.add_argument("--filter_size", default=3, type=int, nargs="?",
                        help="Size of a square filter")
    parser.add_argument("--run_id", default=0, type=int, nargs="?",
                        help="Helps to identify different runs of an experiments")

    args = parser.parse_args()

    # hyperparameters
    lr = args.learning_rate
    num_filters = args.num_filters
    batch_size = args.batch_size
    epochs = args.epochs
    filter_size = args.filter_size

    # train and test convolutional neural network
    x_train, y_train, x_valid, y_valid, x_test, y_test = mnist(args.input_path)

    with tf.Session() as sess:
        training_results, model = train_and_validate(x_train, y_train, x_valid, y_valid, epochs, lr, num_filters, batch_size, filter_size)
        test_results = evaluate(x_test, y_test, model, batch_size)

    print("Training results", training_results)
    print("Test results", test_results)
