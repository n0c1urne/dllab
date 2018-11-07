import matplotlib.pyplot as plt
from cnn_mnist import train_and_validate, mnist
import tensorflow as tf

# use backend without necessity of xserver
import matplotlib as mpl
mpl.use('Agg')


# hyperparameters
lr = 0.1
num_filters = 16
batch_size = 32
epochs = 10

# train and test convolutional neural network
x_train, y_train, x_valid, y_valid, x_test, y_test = mnist('./')

with tf.Session() as sess:

    plt.figure()

    best_accuracies = []

    # train with different filter sizes and plot validation accuracy history
    for filter_size in [1, 3, 5, 7]:
        results, model = train_and_validate(x_train, y_train, x_valid, y_valid, epochs, lr, num_filters, batch_size, filter_size)
        plt.plot(results["val_acc"])
        best_accuracies.append(results["val_acc"][-1])

    # produce image
    plt.xlabel("epoch")
    plt.ylabel("validation accuracy")
    plt.legend(["1", "3", "5", "7"], loc=4)
    plt.savefig("filter_sizes.png")
    print("Best validation accuracies", best_accuracies)
