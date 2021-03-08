from layers import *


class CNN(object):
    """
    Implements Convolutional Neural Network
    Input shape: [8, 3, 32, 32]---------->[batch size, channels, height, width]
    Model Architecture:
    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1             [8, 8, 32, 32]             224
                  ReLU-2             [8, 8, 32, 32]               0
             MaxPool2d-3             [8, 8, 16, 16]               0
                Conv2d-4            [8, 16, 16, 16]           1,168
                  ReLU-5            [8, 16, 16, 16]               0
             MaxPool2d-6              [8, 16, 8, 8]               0
                Linear-7                   [8, 100]         102,500
                  ReLU-8                   [8, 100]               0
                Linear-9                     [8, 3]             303
    ================================================================
    Total params: 104,195
    Trainable params: 104,195
    Non-trainable params: 0
    """
    def __init__(self, in_channels, out_dims):
        self.in_channels = in_channels
        self.out_dims = out_dims

        # C1
        self.conv1 = Conv2d(in_channels=self.in_channels, out_channels=8, kernel_size=3, strides=1)
        self.relu1 = ReLU()
        self.max_pool1 = MaxPooling2d(kernel_size=2, strides=2)

        # C2
        self.conv2 = Conv2d(in_channels=8, out_channels=16, kernel_size=3, strides=1)
        self.relu2 = ReLU()
        self.max_pool2 = MaxPooling2d(kernel_size=2, strides=2)

        self.flatten = Flatten()
        self.fc1 = Dense(in_dims=16 * 8 * 8, out_dims=100)
        self.relu3 = ReLU()
        self.fc2 = Dense(in_dims=100, out_dims=self.out_dims)

        self.softmax = function.softmax

        self.layers = [self.conv1, self.conv2, self.fc1, self.fc2]

    def forward(self, x):
        # C1
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.max_pool1.forward(x)
        # print(x.shape)

        # C2
        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.max_pool2.forward(x)
        # print(x.shape)

        # Flatten
        x = self.flatten.forward(x)
        # print(x.shape)

        # Fully connected layer
        x = self.fc1.forward(x)

        x = self.relu3.forward(x)

        # print(x.shape)
        x = self.fc2.forward(x)
        # print(x.shape)
        output = self.softmax(x)
        # print(x.shape)

        return output

    def backward(self, y, p_y):
        deltaL = p_y - y

        deltaL = self.fc2.backward(deltaL)

        deltaL = self.relu3.backward(deltaL)

        deltaL = self.fc1.backward(deltaL)

        deltaL = self.flatten.backward(deltaL)

        # C2
        deltaL = self.max_pool2.backward(deltaL)
        deltaL = self.relu2.backward(deltaL)
        deltaL = self.conv2.backward(deltaL)

        # C1
        deltaL = self.max_pool1.backward(deltaL)
        deltaL = self.relu1.backward(deltaL)
        self.conv1.backward(deltaL)

    def params(self):
        params = {}
        for i, layer in enumerate(self.layers):
            params['w' + str(i+1)] = layer.params['w']
            params['b' + str(i+1)] = layer.params['b']

        return params

    def set_params(self, params):
        for i, layer in enumerate(self.layers):
            layer.params['w'] = params['w' + str(i+1)]
            layer.params['b'] = params['b' + str(i+1)]

