import torch
import torch.nn as nn

class DICNN(nn.Module):
    def __init__(self, spectral_num, channel=64, reg=True):
        super(DICNN, self).__init__()

        self.reg = reg
        # ConvTranspose2d: output = (input - 1)*stride + outpading - 2*padding + kernelsize
        self.conv1 = nn.Conv2d(in_channels=spectral_num + 1, out_channels=channel, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=channel, out_channels=spectral_num, kernel_size=3, stride=1, padding=1,
                               bias=True)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):

        [lms, pan, ms] = input
        
        # x= lms; y = pan
        x = lms
        y = pan

        input1 = torch.cat([x, y], 1)  # Bsx9x64x64

        rs = self.relu(self.conv1(input1))
        rs = self.relu(self.conv2(rs))
        out = self.conv3(rs)
        output = x + out

        return output


