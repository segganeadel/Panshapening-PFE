import torch
import torch.nn as nn
import torch.nn.functional as F
from metrics_torch.ERGAS_TORCH import ergas_torch
from metrics_torch.SAM_TORCH import sam_torch
try:
    import lightning as L
except:
    import pytorch_lightning as L



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

    def training_step(self, batch, batch_idx):
        y_hat = self(batch)

        y = batch['gt']
        loss = torch.nn.functional.mse_loss(y_hat, y)
        with torch.no_grad():
            ergas = ergas_torch(y_hat, y) 
            sam = sam_torch(y_hat, y)
            self.log_dict({'training_loss': loss, 
                        'training_sam':   sam, 
                        'training_ergas': ergas}, 
                            prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        y_hat = self(batch)

        y = batch['gt']
        loss = torch.nn.functional.mse_loss(y_hat, y)
        with torch.no_grad():
            ergas = ergas_torch(y_hat, y)  
            sam = sam_torch(y_hat, y)
            self.log_dict({'validation_loss':  loss, 
                        'validation_sam':   sam, 
                        'validation_ergas': ergas}, 
                            prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        y_hat = self(batch)

        y = batch['gt']
        loss = torch.nn.functional.mse_loss(y_hat, y)

        with torch.no_grad():
            sam = sam_torch(y_hat, y)
            ergas = ergas_torch(y_hat, y)  
            self.log_dict({'test_loss':  loss, 
                        'test_sam':   sam, 
                        'test_ergas': ergas}, 
                            prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x = batch
        preds = self(x)
        return preds
    