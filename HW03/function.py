import numpy as np
import matplotlib.pyplot as plt


def shuffle(x, y, samples):
    """
    Shuffle input lists data
    :param y: label list
    :param x: image list
    :param samples: number of input data
    :return: shuffle data
    """
    shuffle_index = np.random.permutation(samples)
    shuffle_x, shuffle_y = x[shuffle_index], y[shuffle_index]
    return shuffle_x, shuffle_y


def get_indices(input_shape, output_shape, kernel_size, stride):
    """
    Compute the index matrices to transform input image into a matrix.
    :param input_shape: input image shape
    :param output_shape: output image shape
    :param kernel_size: filter size
    :param stride: int value
    :return: matrix of index
    """

    # get input size
    m, n_C, n_H, n_W = input_shape

    # get output size
    out_h = output_shape[2]
    out_w = output_shape[3]

    # ----Compute matrix of index i----

    # Level 1 vector.
    level1 = np.repeat(np.arange(kernel_size), kernel_size)
    # Duplicate for the other channels.
    level1 = np.tile(level1, n_C)
    # Create a vector with an increase by 1 at each level.
    everyLevels = stride * np.repeat(np.arange(out_h), out_w)
    # Create matrix of index i at every levels for each channel.
    i = level1.reshape(-1, 1) + everyLevels.reshape(1, -1)

    # ----Compute matrix of index j----

    # Slide 1 vector.
    slide1 = np.tile(np.arange(kernel_size), kernel_size)
    # Duplicate for the other channels.
    slide1 = np.tile(slide1, n_C)
    # Create a vector with an increase by 1 at each slide.
    everySlides = stride * np.tile(np.arange(out_w), out_h)
    # Create matrix of index j at every slides for each channel.
    j = slide1.reshape(-1, 1) + everySlides.reshape(1, -1)

    # ----Compute matrix of index d----

    # This is to mark delimitation for each channel
    # during multi-dimensional arrays indexing.
    d = np.repeat(np.arange(n_C), kernel_size * kernel_size).reshape(-1, 1)

    return i, j, d


def create_mask_from_window(x):
    """
    Creates a mask from an input matrix x, to identify the max entry of x.
    :param x: numpy array of shape (kernel size, kernel size)
    :return: mask, Array of the same shape as window, contains a True at the position corresponding to the
             max entry of x.
    """
    mask = (x == np.max(x))
    return mask


def softmax(x):
    """
    Compute the softmax of vector x.
    :param x: input feature, of shape (batch size, number of class)
    :return: numpy array
    """
    exps = np.exp(x)

    return exps / np.sum(exps, axis=1)[:, np.newaxis]


def cross_entropy_loss(y_true, y_pred):
    """
    Implement cross entropy loss function.
    y_true: ground truth, of shape (number of class, batch size)
    y_pred: prediction made by the model, of shape (number of class, batch size)
    """
    L_sum = np.sum(np.multiply(y_true, np.log(y_pred)))
    # batch size
    m = y_true.shape[0]
    # The final loss take the average.
    loss = -(1. / m) * L_sum
    return loss


def copy_weights_to_zeros(params):
    result = {}
    result.keys()
    for key in params.keys():
        result[key] = np.zeros_like(params[key])
    return result


def compute_accuracy(output, y_truth):
    """
        This function does a forward pass of x, then checks if the indices
        of the maximum value in the output equals the indices in the label
        y. Then it sums over each prediction and calculates the accuracy.
    """
    length = len(output)
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
