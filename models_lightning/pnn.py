import torch
import torch.nn as nn
import lightning as L
import torch.nn.functional as F
from metrics_torch.ERGAS_TORCH import ergas_torch
from metrics_torch.SAM_TORCH import sam_torch


class PNN(L.LightningModule):

    def __init__(self, spectral_num, channel=64):
        super(PNN, self).__init__()
        self.spectral_num = spectral_num
        # ConvTranspose2d: output = (input - 1)*stride + outpading - 2*padding + kernelsize
        self.conv1 = nn.Conv2d(in_channels=spectral_num + 1,    out_channels=channel,       kernel_size=9, stride=1)
        self.conv2 = nn.Conv2d(in_channels=channel,             out_channels=32,            kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(in_channels=32,                  out_channels=spectral_num,  kernel_size=5, stride=1)
        self.relu =  nn.ReLU(inplace=True)
        # init_weights(self.conv1, self.conv2, self.conv3)


    def forward(self, input: dict) -> torch.Tensor:  

        lms = input["lms"]
        pan = input["pan"]

        x = torch.cat([lms, pan], 1)
        pad = 8 # (9 - 1) + (5 - 1) + (5 - 1) // 2 = 8    
        x = F.pad(x, (pad, pad, pad, pad), mode='reflect')
        
        input1 = x  # Bsx9x64x64
        rs = self.relu(self.conv1(input1))
        rs = self.relu(self.conv2(rs))
        output = self.conv3(rs)

        return output
    
    def configure_optimizers(self):
        lr = 0.0001 * 17 * 17 * self.spectral_num
        return torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9) 
    
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
    