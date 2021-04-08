import MNIST
import function
import numpy as np
from model import FNN
import time


# Data preprocessing
# Load training data
x, y, examples_train = MNIST.training_data()

# Load testing data
test_x, test_y, examples_test = MNIST.testing_data()
test_x = test_x.T

# one-hot encoding
digits = 10
y = y.reshape(1, examples_train)
test_y = test_y.reshape(1, examples_test)
y = np.eye(digits)[y.astype('int32')]
test_y = np.eye(digits)[test_y.astype('int32')]
y = y.T.reshape(digits, examples_train)
test_y = test_y.T.reshape(digits, examples_test)

# Spilt training data: 70% for training, 30% for validation
spilt = int(0.7 * examples_train)
train_x = x[:spilt].T
train_y = y[:, :spilt]
val_x = x[spilt:].T
val_y = y[:, spilt:]

# Normalize data
train_x = train_x / 255.
val_x = val_x / 255.
test_x = test_x / 255.

# Hyper parameters
input_size = 28 * 28
lr = 0.1
epoch = 101
batch_size = 64

# Initialize model
model = FNN(input_dim=input_size, output_dim=10)


# Training
def train(epochs):
    losses = {'Training_loss': [], 'Validation_loss': [], 'Testing_loss': []}
    accuracies = {'Training_acc': [], 'Validation_acc': [], 'Testing_acc': []}
    step = []
    best_accuracy = 0.
    print("Start training!")
    for i in range(epochs):
        step.append(i+1)
        # Set the learning rate
        model.learning_rate = function.lr_scheduler(i, lr)

        # shuffle training set
        X_train_shuffled, Y_train_shuffled = function.shuffle(train_x, train_y, int(spilt))

        start_time = time.time()
        for j in range(int(spilt/batch_size)+1):
            # get mini-batch
            begin = j * batch_size
            end = min(begin + batch_size, train_x.shape[1] - 1)
            X = X_train_shuffled[:, begin:end]
            Y = Y_train_shuffled[:, begin:end]
            m_batch = end - begin

            output = model.forward(X)
            gradients = model.backward(Y, output, m_batch)
            model.optimize(gradients)

        # Training step
        training_output = model.forward(train_x)
        train_loss = function.cross_entropy_loss(train_y, training_output)
        training_accuracy = function.compute_accuracy(training_output, train_y)
        losses['Training_loss'].append(train_loss)
        accuracies['Training_acc'].append(training_accuracy)

        # Validation step
        val_output = model.forward(val_x)
        val_last_layer_output = model.params['z1']
        val_loss = function.cross_entropy_loss(val_y, val_output)
        val_accuracy = function.compute_accuracy(val_output, val_y)
        losses['Validation_loss'].append(val_loss)
        accuracies['Validation_acc'].append(val_accuracy)

        # Testing step
        testing_output = model.forward(test_x)
        test_last_layer_output = model.params['z1']
        test_loss = function.cross_entropy_loss(test_y, testing_output)
        testing_accuracy = function.compute_accuracy(testing_output, test_y)
        losses['Testing_loss'].append(test_loss)
        accuracies['Testing_acc'].append(testing_accuracy)

        print('Epoch: {0} | Time: {1:.2f}s | Training loss: {2:.4f}, accuracy: {3:.3f}% | '
              'Val loss: {4:.4f}, accuracy: {5:.3f}% | Testing loss: {6:.4f}, accuracy: {7: .3f}%'.format(
            i + 1, time.time() - start_time, train_loss, training_accuracy * 100.,
            val_loss, val_accuracy * 100., test_loss, testing_accuracy * 100.))

        # According to testing accuracy, save the best model.
        if testing_accuracy > best_accuracy:
            best_accuracy = testing_accuracy
            print("Best testing accuracy:{0:.3f}%".format(best_accuracy * 100.))
            print("Save the best model")
            model.save_parameters()

        # Plot t-SNE result, this step will take a long time.
        if i % 50 == 0:
            print("t-SNE Step:")
            function.plot_with_labels(val_last_layer_output, val_y,
                                      str='Visualize validation result at epoch: {0}'.format(i))
            function.plot_with_labels(test_last_layer_output, test_y,
                                      str='Visualize testing result at epoch: {0}'.format(i))

    print("Training finished")
    return losses, accuracies, step


if __name__ == '__main__':
    loss, accuracy, step = train(epoch)
    function.plot_learning_curve(string='Loss', step=step, data=loss, x=epoch)
    function.plot_learning_curve(string='Accuracy', step=step, data=accuracy, x=epoch)
