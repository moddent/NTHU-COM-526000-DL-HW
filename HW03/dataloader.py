import os
import random
import numpy as np
import matplotlib.pyplot as plt


def data_set(folder_path):
    """
    Load image data and return numpy array of image and label.
    class       label
    Carambula     0
    Lychee        1
    Pear          2
    """
    all_dir = os.listdir(folder_path)

    # Save each class data to each list.
    label0_data = []
    label1_data = []
    label2_data = []
    for dir in all_dir:
        for file in os.listdir(folder_path + '/' + str(dir)):
            img_path = folder_path + '/' + str(dir) + '/' + file
            img = plt.imread(img_path)
            img = img[:, :, 0:3]
            if dir == all_dir[0]:
                label0_data.append((img, 0))
            elif dir == all_dir[1]:
                label1_data.append((img, 1))
            else:
                label2_data.append((img, 2))

    # Shuffle each class data.
    random.shuffle(label0_data)
    random.shuffle(label1_data)
    random.shuffle(label2_data)

    # Save each class data to an array.
    imgs = []
    labels = []
    for data in zip(label0_data, label1_data, label2_data):
        for x, y in data:
            imgs.append(x)
            labels.append(y)
    imgs = np.array(imgs)
    imgs = np.reshape(imgs, (imgs.shape[0], imgs.shape[3], imgs.shape[1], imgs.shape[2]))
    labels = np.array(labels)

    return imgs, labels
