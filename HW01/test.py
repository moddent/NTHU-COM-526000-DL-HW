"""
Load the best model parameters.
Test the best model.
"""
import Training
import function
from model import FNN

# Data preprocessing
# Load training data
train_x = Training.train_x
train_y = Training.train_y

# Load validation data
val_x = Training.val_x
val_y = Training.val_y

# Load testing data
test_x = Training.test_x
test_y = Training.test_y

# Hyper parameters
input_size = 28 * 28

# Initialize model
model = FNN(input_dim=input_size, output_dim=10)
model.load_parameters()

print("Start testing!")
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
print("Testing finished")
print('Training loss: {0:.4f}, accuracy: {1:.3f}% | Validation loss: {2:.4f}, accuracy: {3:.3f}% '
      '| Testing loss: {4:.4f} , accuracy: {5:.3f}%'.format(train_loss, train_accuracy * 100., val_loss,
                                                            val_accuracy * 100., test_loss, test_accuracy * 100.))
