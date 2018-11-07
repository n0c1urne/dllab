import matplotlib.pyplot as plt
from cnn_mnist import train_and_validate, mnist
import tensorflow as tf

# use backend without necessity of xserver
import matplotlib as mpl
mpl.use('Agg')


# hyperparameters
num_filters = 16
batch_size = 32
epochs = 10
filter_size = 3

# train and test convolutional neural network
x_train, y_train, x_valid, y_valid, x_test, y_test = mnist('./')

with tf.Session() as sess:

    plt.figure()

    best_accuracies = []

    # train with different learning rates and plot  validation accuracy history
    for lr in [0.1, 0.01, 0.001, 0.0001]:
        results, model = train_and_validate(x_train, y_train, x_valid, y_valid, epochs, lr, num_filters, batch_size, filter_size)
        plt.plot(results["val_acc"])
        best_accuracies.append(results["val_acc"][-1])

    # produce image
    plt.xlabel("epoch")
    plt.ylabel("validation accuracy")
    plt.legend(["0.1", "0.01", "0.001", "0.0001"], loc=4)
    plt.savefig("learning_rates.png")

    print("Best validation accuracies", best_accuracies)
