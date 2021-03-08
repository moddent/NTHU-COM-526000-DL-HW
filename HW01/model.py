import numpy as np
import function


class FNN:
    """
    Implement fully connected network with 3 layers.
    Input layer: 784 nodes
    Hidden layer: 300 nodes
    Output layer: 10 nodes
    """
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Initialize weights and biases for each layer.
        self.hidden_layer_params = function.ini_parameter(input_dim=self.input_dim, output_dim=300)
        self.output_layer_params = function.ini_parameter(input_dim=300, output_dim=self.output_dim)

        # ReLU activation function
        self.relu = function.relu

        # Softmax classifier
        self.classifier = function.softmax

        # Save the each layer parameters
        self.params = {'w1': self.hidden_layer_params['w'], 'b1': self.hidden_layer_params['b'],
                       'w2': self.output_layer_params['w'], 'b2': self.output_layer_params['b']}

        # Set the learning rate
        self.learning_rate = 0.1

    def forward(self, x):
        """
        Implement forward pass.
        :param x: input feature
        :return: probability for each class
        """

        # Input layer
        self.params['a0'] = x

        # Input layer -> hidden layer, z1 = w1 * a0 + b1
        self.params['z1'] = np.matmul(self.params['w1'], self.params['a0']) + self.params['b1']

        # ReLU layer, a1 = ReLU(z1)
        self.params['a1'] = self.relu(self.params['z1'])

        # Hidden layer -> output layer, z2 = w2 * a1 + b2
        self.params['z2'] = np.matmul(self.params['w2'], self.params['a1']) + self.params['b2']

        # Output layer, a2 = Softmax(z2)
        self.params['a2'] = self.classifier(self.params['z2'])

        return self.params['a2']

    def backward(self, y, p_y, m_batch):
        """
        Implement the back propagation process and compute the gradients
        :param m_batch: mini batch size
        :param p_y: model predict y
        :param y: ground truth y
        :return: gradients
        """
        gradients = {}
        # d_z2
        error = p_y - y

        # Gradients at last layer
        gradients['d_w2'] = (1. / m_batch) * np.matmul(error, self.params['a1'].T)
        gradients['d_b2'] = (1. / m_batch) * np.sum(error, axis=1, keepdims=True)

        # Back propagate through first layer
        d_a1 = np.matmul(self.params['w2'].T, error)
        d_z1 = d_a1 * self.relu(self.params['z1'], derivative=True)

        # Gradients at first layer
        gradients['d_w1'] = (1. / m_batch) * np.matmul(d_z1, self.params['a0'].T)
        gradients['d_b1'] = (1. / m_batch) * np.sum(d_z1, axis=1, keepdims=True)

        return gradients

    def optimize(self, gradients):
        """
        Update network parameters according to update rule from
        Stochastic Gradient Descent.
           θ = θ - η * ∇J(x, y),
                theta θ:            a network parameter (e.g. a weight w)
                eta η:              the learning rate
                gradient ∇J(x, y):  the gradient of the objective function,
                                    i.e. the change for a specific theta θ
        :param gradients: the gradients for each parameter
        """
        self.params['w1'] -= self.learning_rate * gradients['d_w1']
        self.params['b1'] -= self.learning_rate * gradients['d_b1']
        self.params['w2'] -= self.learning_rate * gradients['d_w2']
        self.params['b2'] -= self.learning_rate * gradients['d_b2']

    def save_parameters(self):
        """
        Save the model parameters.
        :return: None
        """
        np.savez('Parameter/best_model.npz', w1=self.params['w1'], b1=self.params['b1'],
                 w2=self.params['w2'], b2=self.params['b2'])

    def load_parameters(self):
        """
        Load the model parameters.
        :return: None
        """
        parameters = np.load('Parameter/best_model_98.03%.npz')
        self.params['w1'] = parameters['w1']
        self.params['b1'] = parameters['b1']
        self.params['w2'] = parameters['w2']
        self.params['b2'] = parameters['b2']
