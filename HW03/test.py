"""
Load the best model parameters.
Test the best model.
"""
import train
import function
import numpy as np
from model import CNN
import time

# Data preprocessing
# Load training data
train_x = train.train_x
train_y = train.train_y

# Load validation data
val_x = train.val_x
val_y = train.val_y

# Load testing data
test_x = train.test_x
test_y = train.test_y

# Hyper parameters
input_channel = train_x.shape[1]

# Initialize model

# Initialize model
model = CNN(in_channels=input_channel, out_dims=3)
# parameter = np.load('Parameter/best_model.npz')
parameter = np.load('Parameter/best_model99.598%.npz')
model.set_params(parameter)

print("Start testing!")
start = time.time()
# Test on training set
output = model.forward(train_x)
train_loss = function.cross_entropy_loss(train_y, output)
train_accuracy = function.compute_accuracy(output, train_y)

# Test on validation set
output = model.forward(val_x)
val_loss = function.cross_entropy_loss(val_y, output)
val_accuracy = function.compute_accuracy(output, val_y)

# Test on testing set
output = model.forward(test_x)
test_loss = function.cross_entropy_loss(test_y, output)
test_accuracy = function.compute_accuracy(output, test_y)

print("Testing finished(time:{0:.2f})".format(time.time()-start))
print('Training loss: {0:.4f}, accuracy: {1:.3f}% | Validation loss: {2:.4f}, accuracy: {3:.3f}% '
      '| Testing loss: {4:.4f} , accuracy: {5:.3f}%'.format(train_loss, train_accuracy * 100., val_loss,
                                                            val_accuracy * 100., test_loss, test_accuracy * 100.))
