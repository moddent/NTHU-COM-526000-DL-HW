import gzip
import numpy as np


def training_data():
    """
    Load training data.
    :return: training images and training labels
    """
    # open gzip file
    f = gzip.open("MNIST/train-images-idx3-ubyte.gz", "r")

    # first 4 bytes is a magic number
    f.read(4)

    # second 4 bytes is the number of images
    image_count = int.from_bytes(f.read(4), 'big')

    # third 4 bytes is the row count
    image_height = int.from_bytes(f.read(4), 'big')

    # fourth 4 bytes is the column count
    image_width = int.from_bytes(f.read(4), 'big')

    # rest is the image pixel data, each pixel is stored as an unsigned byte
    # pixel values are 0 to 255
    # read image data
    image_data = f.read()

    # image data to numpy array
    images = np.frombuffer(image_data, dtype=np.uint8).astype(np.float32)

    # resize numpy array size to (number of image, channel, image height, image width)
    images = images.reshape(image_count, image_height * image_width)

    f = gzip.open("MNIST/train-labels-idx1-ubyte.gz", "r")
    # first 4 bytes is a magic number
    f.read(4)

    # second 4 bytes is the number of labels
    label_count = int.from_bytes(f.read(4), 'big')

    # rest is the label data, each label is stored as unsigned byte
    # label values are 0 to 9
    label_data = f.read()
    labels = np.frombuffer(label_data, dtype=np.uint8)
    labels = labels.reshape(-1, 1)
    f.close()
    return images, labels, label_count


def testing_data():
    """
    Load testing data.
    :return: testing images and testing labels
    """
    f = gzip.open("MNIST/t10k-images-idx3-ubyte.gz", "r")
    # first 4 bytes is a magic number
    f.read(4)

    # second 4 bytes is the number of images
    image_count = int.from_bytes(f.read(4), 'big')

    # third 4 bytes is the row count
    image_height = int.from_bytes(f.read(4), 'big')

    # fourth 4 bytes is the column count
    image_width = int.from_bytes(f.read(4), 'big')

    # rest is the image pixel data, each pixel is stored as an unsigned byte
    # pixel values are 0 to 255
    # read image data
    image_data = f.read()

    # image data to numpy array
    images = np.frombuffer(image_data, dtype=np.uint8).astype(np.float32)

    # resize numpy array size to (number of image, channel, image height, image width)
    images = images.reshape(image_count, image_height * image_width)

    f = gzip.open("MNIST/t10k-labels-idx1-ubyte.gz", "r")
    # first 4 bytes is a magic number
    f.read(4)

    # second 4 bytes is the number of labels
    label_count = int.from_bytes(f.read(4), 'big')

    # rest is the label data, each label is stored as unsigned byte
    # label values are 0 to 9
    label_data = f.read()
    labels = np.frombuffer(label_data, dtype=np.uint8)
    labels = labels.reshape(-1, 1)
    f.close()
    return images, labels, label_count
