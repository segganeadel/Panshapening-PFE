import torch
import torch.nn as nn
import cv2


class Resblock(nn.Module):
    def __init__(self):
        super(Resblock, self).__init__()

        channel = 32
        self.conv20 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1,
                                bias=True)
        self.conv21 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1,
                                bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # x= hp of ms; y = hp of pan
        rs1 = self.relu(self.conv20(x))  # Bsx32x64x64
        rs1 = self.conv21(rs1)  # Bsx32x64x64
        rs = torch.add(x, rs1)  # Bsx32x64x64
        return rs

class PanNet(nn.Module):
    def __init__(self, spectral_num, channel=32, reg=True):
        super(PanNet, self).__init__()
        self.reg = reg

        # ConvTranspose2d: output = (input - 1)*stride + outpading - 2*padding + kernelsize
        self.deconv = nn.ConvTranspose2d(in_channels=spectral_num, out_channels=spectral_num, kernel_size=8, stride=4,
                                         padding=2, bias=True)
        self.conv1 = nn.Conv2d(in_channels=spectral_num + 1, out_channels=channel, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.res1 = Resblock()
        self.res2 = Resblock()
        self.res3 = Resblock()
        self.res4 = Resblock()

        self.conv3 = nn.Conv2d(in_channels=channel, out_channels=spectral_num, kernel_size=3, stride=1, padding=1,
                               bias=True)

        self.relu = nn.ReLU(inplace=True)

        self.backbone = nn.Sequential(  # method 2: 4 resnet repeated blocks
            self.res1,
            self.res2,
            self.res3,
            self.res4
        )

    def forward(self, input):# x= hp of ms; y = hp of pan
        
        [lms, pan, ms] = input

        x = ms 
        y = pan 
        
        output_deconv = self.deconv(x)
        input = torch.cat([output_deconv, y], 1)  # Bsx9x64x64
        rs = self.relu(self.conv1(input))  # Bsx32x64x64

        rs = self.backbone(rs)  # ResNet's backbone!

        output = self.conv3(rs) # Bsx8x64x64
        return output