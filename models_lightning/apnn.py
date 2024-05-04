import torch
import torch.nn as nn
import lightning as L

class APNN(L.LightningModule):
    def __init__(self, spectral_num):
        super(APNN, self).__init__()

        channel = 64
        # ConvTranspose2d: output = (input - 1)*stride + outpading - 2*padding + kernelsize

        self.conv1 = nn.Conv2d(in_channels=spectral_num + 1,    out_channels=channel, kernel_size=9, stride=1)
        self.conv2 = nn.Conv2d(in_channels=channel,                  out_channels=32, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(in_channels=32,             out_channels=spectral_num, kernel_size=5, stride=1)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, input):  # x= lms; y = pan
        
        lms = input["lms"]
        pan = input["pan"]

        x = torch.cat([lms, pan], 1)  # BsxCx64x64

        x = torch.nn.functional.pad(x, (8, 8, 8, 8), mode='reflect')
        
        # input1 = self.bn(input1)
        rs = self.relu(self.conv1(x))
        rs = self.relu(self.conv2(rs))  
        output = self.conv3(rs) 

        output = torch.add(output,lms)

        return output

    