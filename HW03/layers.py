import numpy as np
import function


class Conv2d(object):
    """
    Implements the 2D convolutional layer.
    """
    def __init__(self, in_channels, out_channels, kernel_size, strides=1, padding=(0, 0), num_pads=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.num_pads = num_pads
        w, b = self.init_parameters()
        self.params = {'w': w, 'b': b}
        self.grads = {'dw': np.zeros(w.shape), 'db': np.zeros(b.shape)}

        self.cache = None

    def forward(self, x):
        """
        Implements the forward pass for a convolutional layer.
        :param x: input numpy array features with shape: (batch_size, number of channels, Height, Width)
        :return: output numpy array: Z, shape: (batch_size, number of filter, n_H, n_W)
        """
        # Get input size
        batch_size, num_channels, H, W = x.shape

        # Compute the dimensions of the output height and width
        n_H = int((H - self.kernel_size + 2 * self.num_pads) / self.strides) + 1
        n_W = int((W - self.kernel_size + 2 * self.num_pads) / self.strides) + 1

        # Initialize the output Z with zeros
        Z = np.zeros((batch_size, self.out_channels, n_H, n_W))

        # Create x_pad by padding x
        x_pad = self.pad(x, self.num_pads, self.padding)

        # Convolution step
        index_i, index_j, index_d = function.get_indices(x.shape, Z.shape, self.kernel_size, self.strides)
        cols = x_pad[:, index_d, index_i, index_j]
        x_cols = np.concatenate(cols, axis=-1)
        w_col = np.reshape(self.params['w'], (self.out_channels, -1))
        b_col = np.reshape(self.params['b'], (-1, 1))

        # Perform matrix multiplication
        output = np.matmul(w_col, x_cols) + b_col

        # Reshape back matrix to image.
        output = np.array(np.hsplit(output, batch_size)).reshape((batch_size, self.out_channels, n_H, n_W))

        # Final check the out size.
        assert (output.shape == (batch_size, self.out_channels, n_H, n_W))
        Z = output

        # Save the x, x_cols, w_col for backward
        self.cache = x, x_cols, w_col
        return Z

    def init_parameters(self):
        """
        Initialize parameters with Xavier initialization. Sets a layer’s parameters to values chosen from a random
        uniform distribution.
        :return: weights, shape: (out_channels, in_channels, kernel_size, kernel_size)
                 biases, shape: (out_channels,)
        """
        bound = 1 / np.sqrt(self.kernel_size * self.kernel_size)
        weights = np.random.uniform(-bound, bound,
                                    size=(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        biases = np.random.uniform(-bound, bound, size=self.out_channels)
        return weights, biases

    def pad(self, x, n, padding):
        """
        Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image.
        :param x: numpy array of shape (batch_size, n_C, n_H, n_W)
        :param n: integer, amount of padding around each image on vertical and horizontal dimensions
        :param padding: a tuple of padding value
        :return: X_pad -- padded image of shape (batch_size, n_C, n_H + 2*n, n_W + 2*n)
        """
        x_pad = np.pad(x, ((0, 0), (0, 0), (n, n), (n, n)), 'constant', constant_values=padding)
        return x_pad

    def backward(self, dz):
        """
        Implement the backward propagation for a convolutional layer.
        :param dz: gradient of the cost with respect to the output of the conv layer (Z), shape: (batch_size, n_C, n_H, n_W)
        :return: - dX: error of the current convolutional layer.
                 - self.dw: weights gradient.
                 - self.db: bias gradient.
        """
        # Get the output of previous layer
        x, x_cols, w_col = self.cache
        output_shape = dz.shape

        # Initialize dx
        dx = np.zeros(x.shape)

        # Pad dx
        dx_pad = self.pad(dx, self.num_pads, self.padding)

        # Get batch size
        batch_size = x.shape[0]

        # Compute bias gradient
        self.grads['db'] = np.sum(dz, axis=(0, 2, 3))

        # Reshape dz properly.
        dz = np.reshape(dz, (dz.shape[0] * dz.shape[1], dz.shape[2] * dz.shape[3]))
        dz = np.array(np.vsplit(dz, batch_size))
        dz = np.concatenate(dz, axis=-1)

        # Perform matrix multiplication between reshaped dz and w_col to get dx_cols.
        # Compute the gradient of previous layer' output
        dx_cols = np.matmul(w_col.T, dz)
        # Compute weight gradient
        dw_col = np.matmul(dz, x_cols.T)

        index_i, index_j, index_d = function.get_indices(x.shape, output_shape, self.kernel_size, self.strides)
        dx_cols_reshaped = np.array(np.hsplit(dx_cols, batch_size))
        # Reshape matrix back to image
        np.add.at(dx_pad, (slice(None), index_d, index_i, index_j), dx_cols_reshaped)
        # Remove padding from new image.
        dx = dx_pad[:, :, self.num_pads:-self.num_pads, self.num_pads:-self.num_pads]

        # Reshape dw_col into dw.
        self.grads['dw'] = np.reshape(dw_col, (self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))

        #  Final check the output shape is correct
        assert (dx.shape == x.shape)

        return dx


class MaxPooling2d(object):
    """
    Implent 2D max pooling layer.
    """
    def __init__(self, kernel_size, strides):
        self.kernel_size = kernel_size
        self.strides = strides
        self.cache = None

    def forward(self, x):
        """
        Implements the forward pass of the Max pooling layer.
        :param x: input feature, shape: (batch_size, number of channels, Height, Width)
        :return: a numpy array of shape (batch_size, n_C, n_H, n_W).
        """
        # Save the x for backward
        self.cache = x

        # Get the input shape of x
        batch_size, num_channels, H, W = x.shape

        # Compute the dimensions of the CONV output volume
        n_H = int((H - self.kernel_size) / self.strides) + 1
        n_W = int((W - self.kernel_size) / self.strides) + 1
        n_C = num_channels

        # Initialize output matrix A
        A = np.zeros((batch_size, n_C, n_H, n_W))

        # Pooling step
        for i in range(batch_size):  # loop over the batch size
            for c in range(n_C):  # loop on the vertical axis of the output volume

                for h in range(n_H):  # loop on the horizontal axis of the output volume
                    vert_top = h * self.strides
                    vert_bottom = vert_top + self.kernel_size

                    for w in range(n_W):  # loop over the channels of the output volume
                        horiz_left = w * self.strides
                        horiz_right = horiz_left + self.kernel_size

                        # Find the corners of the current "slice".
                        slice_map = x[i, c, vert_top:vert_bottom, horiz_left:horiz_right]

                        # Compute the max pooling operation on the slice. Use np.max
                        A[i, c, h, w] = np.max(slice_map)
        # Final check the output shape is correct
        assert(A.shape == (batch_size, n_C, n_H, n_W))

        return A

    def backward(self, dz):
        """
        Implements the backward pass of the max pooling layer
        :param dz: gradient of cost with respect to the output of the max pooling layer
        :return: dx, gradient of cost with respect to the input of the pooling layer
        """
        # Get the previous layer's output
        x = self.cache

        # Get dz shape
        batch_size, n_C, n_H, n_W = dz.shape

        # Initialize dx with zeros
        dx = np.zeros(x.shape)

        for i in range(batch_size):  # loop over the batch size

            for c in range(n_C):
                for h in range(n_H):
                    vert_top = h * self.strides
                    vert_bottom = vert_top + self.kernel_size
                    for w in range(n_W):

                        horiz_left = w * self.strides
                        horiz_right = horiz_left + self.kernel_size

                        # Use the corners and "c" to define the current slice from slice_map
                        slice_map = x[i, c, vert_top:vert_bottom, horiz_left:horiz_right]

                        # Create the mask from slice_map
                        mask = function.create_mask_from_window(slice_map)

                        # Set dx to be dx + (the mask multiplied by the correct entry of dx)
                        dx[i, c, vert_top:vert_bottom, horiz_left:horiz_right] += np.multiply(mask, dz[i, c, h, w])
            # Final check the output shape is correct
            assert (dx.shape == x.shape)

        return dx


class Flatten(object):
    """
    Reshape the input feature
    """
    def __init__(self):
        self.forward_shape = None

    def forward(self, x):
        self.forward_shape = x.shape
        x_flatten = np.reshape(x, (self.forward_shape[0], -1))
        return x_flatten

    def backward(self, dz):
        dz = np.reshape(dz, self.forward_shape)
        return dz


class Dense(object):
    """
    Implement fully connected layer.
    """
    def __init__(self, in_dims, out_dims):
        self.in_dims = in_dims
        self.out_dims = out_dims
        w, b = self.init_parameters()
        self.params = {'w': w, 'b': b}
        self.grads = {'dw': np.zeros(self.params['w'].shape),
                      'db': np.zeros(self.params['b'].shape)}
        self.cache = None

    def init_parameters(self):
        weights = np.random.randn(self.out_dims, self.in_dims) * np.sqrt(1. / self.in_dims)
        biases = np.random.randn(1, self.out_dims) * np.sqrt(1. / self.in_dims)
        return weights, biases

    def forward(self, x):
        """
        Implement forward pass for fully connected layer.
        :param x: input feature, shape: (batch_size, in_dims)
        :return: out, shape: (batch_size, out_dims)
        """
        # Save the x for backward
        self.cache = x

        z = np.matmul(x, self.params['w'].T) + self.params['b']

        return z

    def backward(self, dz):
        """
        Implements the backward pass for fully connected layer
        :param dz: gradient of cost with respect to the output of the max pooling layer
        :return: dx, gradient of cost with respect to the input of the pooling layer
        """
        x = self.cache
        batch_size = x.shape[0]

        self.grads['dw'] = (1. / batch_size) * np.matmul(dz.T, x)
        self.grads['db'] = (1. / batch_size) * np.sum(dz, axis=0, keepdims=True)

        dx = np.matmul(dz, self.params['w'])

        return dx


class ReLU(object):
    """
    Implement:
    ReLU activation function,
    ReLU(x)=max(0,x).
    Derivative of ReLU,
    1 for x >= 0, 0 for x < 0.
    """
    def __init__(self):
        self.cache = None

    def forward(self, x):
        self.cache = x
        return np.maximum(0, x)

    def backward(self, dz):
        x = self.cache
        dx = dz * np.where(x <= 0, 0, 1)
        return dx


class SGD(object):
    """
    Update network parameters according to update rule from
    Stochastic Gradient Descent.
      θ = θ - η * ∇J(x, y),
           theta θ:            a network parameter (e.g. a weight w)
           eta η:              the learning rate
           gradient ∇J(x, y):  the gradient of the objective function,
                               i.e. the change for a specific theta θ
    """
    def __init__(self, params, lr, decay=1e-5):
        self.params = function.copy_weights_to_zeros(params)
        self.lr = self.init_lr = lr
        self.iterations = 0
        self.decay = decay

    def step(self, model):
        """
        Update parameters
        :param model: CNN
        """
        # Decay the learning rate every iteration
        self.lr = self.init_lr / (1 + self.iterations * self.decay)

        for layer in model.layers:
            for key in layer.params.keys():
                self.params[key] = self.lr * layer.grads['d' + key]
                layer.params[key] -= self.params[key]

        self.iterations += 1

