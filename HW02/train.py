from functions import *
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from model import Autoencoder


# Hyper parameter
path = 'wafer.zip'
weights_path = './weights/model.pth'
batch_size = 16
epochs = 1000
learning_rate = 1e-2

# Data preprocessing
# Data augmentation
data_transforms = transforms.Compose([transforms.ToTensor(),
                                      AddGaussianNoise(mean=0., std=0.1)])

# Load data
train_data = wafer_dataset(transform=data_transforms, folder_path=path, test_gen=False)
# Set data loader
train_data_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

# Set training device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize model
model = Autoencoder()
model.to(device)

# Set loss function
MSE_loss = nn.MSELoss()
# Set optimizer
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Learning rate scheduler, decay the learning rate every 100 epoch.
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

writer = SummaryWriter()


# training step
def train():
    epoch = []
    losses = []
    print("Start training.")
    for i in range(epochs):
        epoch.append(i+1)
        loss_sum = 0.
        loss_avg = 0.
        model.train()
        for step, (x, _) in enumerate(train_data_loader):
            x = x.to(device, dtype=torch.float)
            outputs = model(x)
            loss = MSE_loss(outputs, x)
            optimizer.zero_grad()
            loss.backward()
            for name, param in model.named_parameters():
                writer.add_histogram(name, param.grad.data.cpu().numpy(), step)
            optimizer.step()
            loss_sum += loss.data.cpu().numpy()
            loss_avg = loss_sum / (step + 1)
            if step % 50 == 0:
                print('Epoch:', i, '| train loss: %.4f' % loss.data.cpu().numpy())
        scheduler.step(epoch=None)
        losses.append(loss_avg)
        writer.close()
        break
    print("Training finished.")
    print("Save the model.")
    model.save_weights(path=weights_path)
    return epoch, losses
# writer.add_histogram_raw()

if __name__ == '__main__':
    epoch, loss = train()
    print("Show training loss curve")
    plot_learning_curve(step=epoch, loss=loss, x=epochs)
