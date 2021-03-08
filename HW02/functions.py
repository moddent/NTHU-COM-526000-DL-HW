import zipfile
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class wafer_dataset(Dataset):
    def __init__(self, transform=None, folder_path=None, test_gen=False):
        """
        :param transform: for augmentation
        :param folder_path: Data set path
        Classes:
        Center (0)
        Donut (1)
        Edge-Loc (2)
        Edge-Ring (3)
        Loc (4)
        Near-full (5)
        Random (6)
        Scratch (7)
        None (8)
        """
        self.classes = ['Center',
                        'Donut',
                        'Edge-Loc',
                        'Edge-Ring',
                        'Loc',
                        'Near-full',
                        'Random',
                        'Scratch',
                        'None']
        data_set = {}
        with zipfile.ZipFile(folder_path) as zf:
            dataset = np.load(folder_path)
            for file_name in zf.namelist():
                data_set[file_name] = dataset[file_name]
        self.test_gen = test_gen
        if not self.test_gen:
            self.data = data_set['data.npy']
            self.label = data_set['label.npy']
        else:
            self.data = data_set['gen_data.npy']
            self.label = data_set['gen_label.npy']
        self.transform = transform

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        if self.transform is not None:
            data = self.transform(data)
        return data, label

    def __len__(self):
        return len(self.data)


def imshow(truth_img, decode_img, class_name, test=False):
    """
    Plot image.
    if test, then plot each class data.
    :param truth_img: Original image
    :param class_name: Wafer map classes
    :param decode_img: A list of generate images
    :param test: for testing
    """
    if not test:
        truth_img = truth_img.cpu().numpy()
        f, axes = plt.subplots(1, 6)
        for i in range(6):
            if i == 0:
                axes[i].set_title('Original image: ' + class_name)
                axes[i].imshow(np.argmax(np.transpose(truth_img, (1, 2, 0)), axis=2))
            else:
                decode_img[i-1] = decode_img[i-1].cpu().numpy()
                axes[i].set_title('Generate image')
                axes[i].imshow(np.argmax(np.transpose(decode_img[i-1], (1, 2, 0)), axis=2))
    else:
        f, axes = plt.subplots(2, 9)
        f.suptitle('Original image', x=0.5, y=0.9, fontsize=20)
        # Adjust vertical_spacing = 0.5 * axes_height
        plt.subplots_adjust(hspace=0.5)
        # Add text in figure coordinates
        plt.figtext(0.5, 0.5, 'Generated image', ha='center', va='center', fontsize=20)
        for i in range(2):
            for j in range(9):
                if i == 0:
                    axes[i, j].set_title(class_name[j])
                    axes[i, j].imshow(np.argmax(np.transpose(truth_img[j], (1, 2, 0)), axis=2))
                else:
                    axes[i, j].set_title(class_name[j])
                    axes[i, j].imshow(np.argmax(np.transpose(decode_img[j], (1, 2, 0)), axis=2))
    plt.show()


def plot_learning_curve(step, loss, x):
    """
    Plot the loss curve for every epoch.
    """
    fig = plt.figure(num=0)
    plt.xlabel('Epoch')
    plt.plot(step, loss, label='training')

    fig.suptitle('Loss' + "\n", fontsize=25)
    plt.axis([1, x, 0, 0.2])
    plt.legend(loc='lower right')
    plt.show()


class AddGaussianNoise(object):
    """
    Add Gaussian Noise to input tensor.
    """
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
