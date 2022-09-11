# Please include your models here
# Please include wrappers around your models, data, and loss functions here
from . import *

class ConvLayer(nn.Module):
    def __init__(
        self,
        inchannels,
        outchannels,
        kernel_size: int=3):
        super(ConvLayer, self).__init__()
        self._inchannels = inchannels
        self._kernel_size = kernel_size
        self.conv1 = nn.Conv2d(
            inchannels, 
            outchannels,
            kernel_size=self._kernel_size
            )
        self.conv2 = nn.Conv2d(
            outchannels,
            outchannels,
            kernel_size=self._kernel_size
            )
        self.net = nn.Sequential(
            self.conv1, 
            nn.ReLU(),
            self.conv2,
            nn.ReLU())
        self._initialize_params()
    
    def __str__(self):
        return "ConvLayer"

    def _initialize_params(self):
        for name, layer in self.named_parameters():
            if "conv" in name:
                if "weight" in name:
                    nn.init.kaiming_normal_(layer, mode='fan_in', nonlinearity='relu')
                elif "bias" in name:
                    nn.init.constant_(layer, 0)

    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, channel_list: Tuple=(1,64,128,256,512,1024,)):
        super(UNet, self).__init__()
        self._convlayers = []
        self.conv1 = ConvLayer(
            channel_list[0],
            channel_list[1])
        self.pool1 = nn.MaxPool2d((2,2), 2)
        self.conv2 = ConvLayer(
            channel_list[1], 
            channel_list[2])
        self.pool2 = nn.MaxPool2d((2,2), 2)
        self.conv3 = ConvLayer(
            channel_list[2], 
            channel_list[3]
            )
        self.pool3 = nn.MaxPool2d((2,2), 2)
        self.conv4 = ConvLayer(
            channel_list[3], 
            channel_list[4]
            )
        self.pool4 = nn.MaxPool2d((2,2), 2)
        self.conv5 = ConvLayer(
            channel_list[4], 
            channel_list[5]
            )        

        self.upsample1 = nn.ConvTranspose2d(channel_list[5], channel_list[4], 2, 2)
        self.conv6 = ConvLayer(
            channel_list[5],
            channel_list[4]
            )
        self.upsample2 = nn.ConvTranspose2d(channel_list[4], channel_list[3], 2,2)
        
        self.conv7 = ConvLayer(
            channel_list[4],
            channel_list[3]
            )
        
        self.upsample3 = nn.ConvTranspose2d(channel_list[3], channel_list[2], 2,2)
        self.conv8 = ConvLayer(
            channel_list[3],
            channel_list[2]
            )
        self.upsample4 = nn.ConvTranspose2d(channel_list[2], channel_list[1], 2,2)
        self.conv9 = ConvLayer(
            channel_list[2],
            channel_list[1]
            )
        self.conv10 = nn.Conv2d(channel_list[1], 1, 1, 1)

    def _initialize_params(self):
        raise NotImplementedError

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(self.pool1(x1))
        x3 = self.conv3(self.pool2(x2))
        x4 = self.conv4(self.pool3(x3))
        x5 = self.conv5(self.pool4(x4))
        
        x_up_1 = self.upsample1(x5)
        x_up_1 = torch.concat((
            transforms.CenterCrop(x_up_1.shape[2:3])(x4),
            x_up_1), dim=1)

        x_up_2 = self.upsample2(self.conv6(x_up_1))
        x_up_2 = torch.concat(
            (transforms.CenterCrop(x_up_2.shape[2:3])(x3),
            x_up_2), dim=1)

        x_up_3 = self.upsample3(self.conv7(x_up_2))
        x_up_3 = torch.concat(
            (transforms.CenterCrop(x_up_3.shape[2:3])(x2),
            x_up_3), dim=1)

        x_up_4 = self.upsample4(self.conv8(x_up_3))
        x_up_4 = torch.concat(
            (transforms.CenterCrop(x_up_4.shape[2:3])(x1),
            x_up_4), dim=1)

        xout = self.conv10(self.conv9(x_up_4))
        return xout