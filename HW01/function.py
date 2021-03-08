import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE


def shuffle(x, y, samples):
    """
    Shuffle input lists data
    :param y: label list
    :param x: image list
    :param samples: number of input data
    :return: shuffle data
    """
    shuffle_index = np.random.permutation(samples)
    shuffle_x, shuffle_y = x[:, shuffle_index], y[:, shuffle_index]
    return shuffle_x, shuffle_y


def ini_parameter(input_dim, output_dim):
    """
    Initialize weights and biases of a layer
    :param input_dim: input dimension
    :param output_dim: output dimension
    :return: weights and biases
    """
    params = {'w': np.random.randn(output_dim, input_dim) * np.sqrt(1. / input_dim),
              'b': np.random.randn(output_dim, 1) * np.sqrt(1. / input_dim)}
    return params


def relu(x, derivative=False):
    """
    Implement:
    ReLU activation function,
    ReLU(x)=max(0,x).
    Derivative of ReLU,
    1 for x >= 0, 0 for x < 0.
    :param x: input feature
    :param derivative: True for backward propagation, else for forward.
    :return: numpy array
    """
    if derivative:
        return np.where(x <= 0, 0, 1)
    else:
        return np.maximum(0, x)


def softmax(x):
    """
    Compute the softmax of vector x.
    :param x: input feature, of shape (number of class, batch size)
    :return: numpy array
    """
    exps = np.exp(x)
    return exps / np.sum(exps, axis=0)


def cross_entropy_loss(y_true, y_pred):
    """
    Implement cross entropy
    y_true: ground truth, of shape (number of class, batch size)
    y_pred: prediction made by the model, of shape (number of class, batch size)
    """
    L_sum = np.sum(np.multiply(y_true, np.log(y_pred)))
    # batch size
    m = y_true.shape[1]
    # The final loss take the average.
    loss = -(1. / m) * L_sum
    return loss


def lr_scheduler(epoch, lr):
    """
    Decay the learning rate every 50 epoch.
    """
    drop = 0.5
    epochs_drop = 25
    lr = lr * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lr


def compute_accuracy(output, y_truth):
    """
        This function does a forward pass of x, then checks if the indices
        of the maximum value in the output equals the indices in the label
        y. Then it sums over each prediction and calculates the accuracy.
    """
    output = output.T
    length = len(output)
    y_truth = y_truth.T
    count = 0
    for p_y, y in zip(output, y_truth):
        pred = np.argmax(p_y)
        y_t = np.argmax(y)
        if pred == y_t:
            count += 1
    return count / length


def plot_learning_curve(string, step, data, x):
    """
    Plot the accuracy curve and loss curve for every epoch.
    """
    fig = plt.figure(num=0)
    plt.xlabel('Epoch')
    if string == 'Loss':
        plt.plot(step, data['Training_loss'], label='training')
        plt.plot(step, data['Validation_loss'], label='validation')
        plt.plot(step, data['Testing_loss'], label='testing')
    else:
        plt.plot(step, data['Training_acc'], label='training')
        plt.plot(step, data['Validation_acc'], label='validation')
        plt.plot(step, data['Testing_acc'], label='testing')
    fig.suptitle(string + "\n", fontsize=25)
    plt.axis([1, x, 0, 1])
    plt.legend(loc='lower right')
    plt.show()


def plot_with_labels(last_layer_output, y, str):
    """
    Visualize last layer feature.
    This function will take a long time, when inputs large batch size of data.
    """
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    # plot_only = 1000
    last_layer_output = last_layer_output.T
    y = y.T
    # y = y[:plot_only]
    labels = []
    for label in y:
        labels.append(np.argmax(label))
    # low_dim_embs = tsne.fit_transform(last_layer_output[:plot_only, :])
    low_dim_embs = tsne.fit_transform(last_layer_output)
    plt.cla()
    X, Y = low_dim_embs[:, 0], low_dim_embs[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9))
        plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.title(str)
    plt.show()
    plt.pause(0.01)
    plt.ioff()
