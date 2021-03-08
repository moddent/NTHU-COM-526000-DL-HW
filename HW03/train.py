import dataloader
import function
import numpy as np
from model import CNN
from layers import SGD
import argparse
import time


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='HW3 Training Script')
parser.add_argument('--epoch', default=10, type=int,
                    help='Epoch for training.')
parser.add_argument('--batch_size', default=8, type=int,
                    help='Batch size for training.')
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float,
                    help='Initial learning rate. Leave as None to read this from the config.')
parser.add_argument('--decay', '--weight_decay', default=1e-5, type=float,
                    help='Weight decay for SGD. Leave as None to read this from the config.')
parser.add_argument('--save_folder', default='Parameter/',
                    help='Directory for saving checkpoint models.')
parser.add_argument('--show_progressbar', default=True, type=str2bool,
                    help=' Whether show progress bar during training.')

args = parser.parse_args()

try:
    from tqdm import trange
except ImportError:
    print('tqdm could not be imported. If you want to use progress bar during training,'
          'install tqdm from https://github.com/tqdm/tqdm.')
    args.show_progressbar = False


# -----------Data preprocessing step ---------
print("Data preprocessing...")
start_time = time.time()
# Load training data
train_path = 'Data_train'
x, y = dataloader.data_set(folder_path=train_path)
train_length = len(y)

# Load testing data
test_path = 'Data_test'
test_x, test_y = dataloader.data_set(folder_path=test_path)
test_length = len(test_y)

# Shuffle the training set
# x, y = function.shuffle(x, y, train_length)

# one-hot encoding
digits = 3
y = y.reshape(1, train_length)
test_y = test_y.reshape(1, test_length)
y = np.eye(digits)[y.astype('int32')]
test_y = np.eye(digits)[test_y.astype('int32')]
y = y.T.reshape(digits, train_length)
test_y = test_y.T.reshape(digits, test_length)
test_y = np.transpose(test_y, (1, 0))

# Spilt training data: 70% for training, 30% for validation
spilt = int(0.7 * train_length)
train_x = x[:spilt]
train_y = y[:, :spilt].T
val_x = x[spilt:]
val_y = y[:, spilt:].T

print("Done!(time:{0:.2f}s)".format(time.time() - start_time))
# ---------------End of data preprocessing---------------

# Hyper parameters
input_channel = train_x.shape[1]
lr = args.lr
decay = args.decay
epoch = args.epoch
batch_size = args.batch_size

# Initialize model
model = CNN(in_channels=input_channel, out_dims=3)
optimizer = SGD(params=model.params(), lr=lr, decay=decay)


# Training
def train(epochs):
    losses = {'Training_loss': [], 'Validation_loss': [], 'Testing_loss': []}
    accuracies = {'Training_acc': [], 'Validation_acc': [], 'Testing_acc': []}
    step = []
    best_accuracy = 0.
    print("Start training!")
    for i in range(epochs):
        step.append(i + 1)

        if args.show_progressbar:
            iterations = trange(int(spilt / batch_size + 1))
            iterations.colour = '#FFFFFF'
        else:
            iterations = range(int(spilt / batch_size + 1))

        # shuffle training set
        X_train_shuffled, Y_train_shuffled = function.shuffle(train_x, train_y, int(spilt))

        for j in iterations:
            # get mini-batch
            begin = j * batch_size
            end = min(begin + batch_size, train_x.shape[0])
            X = X_train_shuffled[begin:end:, ]
            Y = Y_train_shuffled[begin:end:, ]

            output = model.forward(X)
            model.backward(Y, output)
            optimizer.step(model)
            loss = function.cross_entropy_loss(Y, output)
            if args.show_progressbar:
                iterations.set_description("Epoch {0} | loss {1:.3f} |=======>".format(i + 1, loss))
            else:
                print("Epoch {0} | loss {1:.3f}".format(i + 1, loss))

            # Fix bug when batch_size is 1
            if end == train_x.shape[0] and batch_size == 1:
                break

        if args.show_progressbar:
            iterations.close()
        start_time = time.time()
        print('Compute accuracy and loss...')
        # Training step
        training_output = model.forward(train_x)
        train_loss = function.cross_entropy_loss(train_y, training_output)
        training_accuracy = function.compute_accuracy(training_output, train_y)
        losses['Training_loss'].append(train_loss)
        accuracies['Training_acc'].append(training_accuracy)

        # Validation step
        val_output = model.forward(val_x)
        val_loss = function.cross_entropy_loss(val_y, val_output)
        val_accuracy = function.compute_accuracy(val_output, val_y)
        losses['Validation_loss'].append(val_loss)
        accuracies['Validation_acc'].append(val_accuracy)

        # Testing step
        testing_output = model.forward(test_x)
        test_loss = function.cross_entropy_loss(test_y, testing_output)
        testing_accuracy = function.compute_accuracy(testing_output, test_y)
        losses['Testing_loss'].append(test_loss)
        accuracies['Testing_acc'].append(testing_accuracy)
        print('Done!')

        print('Time: {0:.2f}s | Training loss: {1:.4f}, accuracy: {2:.3f}% | '
              'Val loss: {3:.4f}, accuracy: {4:.3f}% | Testing loss: {5:.4f}, accuracy: {6: .3f}%'.format(
            time.time() - start_time, train_loss, training_accuracy * 100.,
            val_loss, val_accuracy * 100., test_loss, testing_accuracy * 100.))

        # According to testing accuracy, save the best model.
        if testing_accuracy > best_accuracy:
            best_accuracy = testing_accuracy
            print("Best testing accuracy:{0:.3f}%".format(best_accuracy * 100.))
            np.savez(args.save_folder + 'best_model.npz', **model.params())
            print("Save the best model")
        print()

    print("Training finished")
    print("Best testing accuracy:{0:.3f}%".format(best_accuracy * 100.))
    return losses, accuracies, step


if __name__ == '__main__':
    loss, accuracy, step = train(epoch)
    function.plot_learning_curve(string='Loss', step=step, data=loss, x=epoch)
    function.plot_learning_curve(string='Accuracy', step=step, data=accuracy, x=epoch)
