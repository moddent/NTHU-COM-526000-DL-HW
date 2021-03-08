import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchsummary import summary


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # Encoder layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)

        # Activation layer, pooling layer and Batch Normalization layer
        self.relu = nn.ReLU(True)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.maxpooling = nn.MaxPool2d(2)

    def forward(self, x):
        encode = self.conv1(x)
        encode = self.relu(encode)
        encode = self.bn1(encode)
        encode = self.maxpooling(encode)
        return encode


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # Decoder layers
        self.deconv1 = nn.ConvTranspose2d(64, 64, 3, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 3, 3, padding=1)

        # Activation layer, pooling layer and Batch Normalization layer
        self.relu = nn.ReLU(True)
        self.bn = nn.BatchNorm2d(num_features=64)

    def forward(self, x):
        decode = self.deconv1(x)
        decode = self.relu(decode)
        decode = self.bn(decode)
        # Upsampling step
        decode = F.interpolate(decode, scale_factor=2, mode='bilinear', align_corners=True)
        decode = self.deconv2(decode)
        x = torch.sigmoid(decode)
        return x


class Autoencoder(nn.Module):
    """
    Model architecture:
    Input Shape:[1, 3, 26, 26]
    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1            [1, 64, 26, 26]           1,792
                  ReLU-2            [1, 64, 26, 26]               0
           BatchNorm2d-3            [1, 64, 26, 26]             128
             MaxPool2d-4            [1, 64, 13, 13]               0
               Encoder-5            [1, 64, 13, 13]               0
       ConvTranspose2d-6            [1, 64, 13, 13]          36,928
                  ReLU-7            [1, 64, 13, 13]               0
           BatchNorm2d-8            [1, 64, 13, 13]             128
       ConvTranspose2d-9             [1, 3, 26, 26]           1,731
              Decoder-10             [1, 3, 26, 26]               0
    ================================================================
    Total params: 40,707
    Trainable params: 40,707
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.01
    Forward/backward pass size (MB): 1.43
    Params size (MB): 0.16
    Estimated Total Size (MB): 1.60
    ----------------------------------------------------------------
    """
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = Encoder()

        # Decoder
        self.decoder = Decoder()

    def forward(self, x):
        latent_vector = self.encoder(x)
        x = self.decoder(latent_vector)
        return x

    def save_weights(self, path):
        """ Saves the model's weights using compression because the file sizes were getting too big. """
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        """ Loads weights from a compressed save file. """
        state_dict = torch.load(path)
        self.load_state_dict(state_dict, strict=False)


if __name__ == '__main__':
    model = Autoencoder()
    print(model)
    # model.cuda()
    # summary(model, (3, 26, 26), batch_size=1)
