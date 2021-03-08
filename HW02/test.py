from functions import *
from torch.utils.data import DataLoader
from torchvision import transforms


# Hyper parameter
ori_path = 'wafer.zip'
gen_path = 'gen_wafer.zip'
batch_size = 1

# Data preprocessing
data_transforms = transforms.Compose([transforms.ToTensor()])
# Load data
ori_data = wafer_dataset(transform=data_transforms, folder_path=ori_path, test_gen=False)
# print(ori_data.data.shape)    # (1281, 26, 26, 3)
gen_data = wafer_dataset(transform=data_transforms, folder_path=gen_path, test_gen=True)
# print(gen_data.data.shape)    # (6405, 26, 26, 3)

# Set data loader
ori_data_loader = DataLoader(dataset=ori_data, batch_size=batch_size, shuffle=False)
gen_data_loader = DataLoader(dataset=gen_data, batch_size=batch_size, shuffle=False)


def get_each_class_data(dataloader):
    """
    Get each class data.
    :param dataloader: Dataloader
    :return: a list of each class and each data.
    """
    classes = []
    ims = []
    for step, (x, y) in enumerate(dataloader):
        class_name = gen_data.classes[y[0].item()]
        if class_name not in classes:
            classes.append(class_name)
            gen_img = x.squeeze(0)
            ims.append(gen_img)
    return classes, ims


# Testing step
def test():
    """
    This function show the original image and generated image for each class.
    """
    print("Start testing!")
    ori_classes, ori_ims = get_each_class_data(dataloader=ori_data_loader)
    _, gen_ims = get_each_class_data(dataloader=gen_data_loader)
    imshow(truth_img=ori_ims, decode_img=gen_ims, class_name=ori_classes, test=True)
    print("Testing finish")


if __name__ == '__main__':
    test()
