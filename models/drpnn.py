import torch
import torch.nn as nn

class Repeatblock(nn.Module):
    def __init__(self):
        super(Repeatblock, self).__init__()

        channel = 32  # input_channel =
        self.conv2 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=7, stride=1, padding=3,
                               bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        rs = self.relu(self.conv2(x))

        return rs

class DRPNN(nn.Module):
    def __init__(self, spectral_num, channel=32):
        super(DRPNN, self).__init__()

        # ConvTranspose2d: output = (input - 1)*stride + outpading - 2*padding + kernelsize
        self.conv1 = nn.Conv2d(in_channels=spectral_num+1, out_channels=channel, kernel_size=7, stride=1, padding=3,
                                  bias=True)

        self.conv2 = nn.Conv2d(in_channels=channel, out_channels=spectral_num+1, kernel_size=7, stride=1, padding=3,
                                  bias=True)
        self.conv3 = nn.Conv2d(in_channels=spectral_num+1, out_channels=spectral_num, kernel_size=7, stride=1, padding=3,
                                  bias=True)
        self.relu = nn.ReLU(inplace=True)

        self.backbone = nn.Sequential(  # method 2: 4 resnet repeated blocks
            Repeatblock(),
            Repeatblock(),
            Repeatblock(),
            Repeatblock(),
            Repeatblock(),
            Repeatblock(),
            Repeatblock(),
            Repeatblock(),
        )

    def forward(self, ms, lms, pan):  # x= lms; y = pan
        
        input = torch.cat([lms, pan], 1)  # Bsx9x64x64
        rs = self.relu(self.conv1(input))  # Bsx64x64x64

        rs = self.backbone(rs)  # backbone!  Bsx64x64x64

        out_res = self.conv2(rs)  # Bsx9x64x64
        output1 = torch.add(input, out_res)  # Bsx9x64x64
        output  = self.conv3(output1)  # Bsx8x64x64

        return output