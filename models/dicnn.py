import torch
import torch.nn as nn
import lightning as L
import torch.nn.functional as F
from metrics_torch.ERGAS_TORCH import ergas_torch
from metrics_torch.SAM_TORCH import sam_torch


class DICNN(L.LightningModule):
    def __init__(self, spectral_num, channel=64, reg=True):
        super(DICNN, self).__init__()

        self.reg = reg
        # ConvTranspose2d: output = (input - 1)*stride + outpading - 2*padding + kernelsize
        self.conv1 = nn.Conv2d(in_channels=spectral_num + 1,    out_channels=channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=channel,             out_channels=channel, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=channel,        out_channels=spectral_num, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input): # x= lms; y = pan
        lms = input["lms"]
        pan = input["pan"]
    
        input1 = torch.cat([lms, pan], 1)  # Bsx9x64x64

        rs = self.relu(self.conv1(input1))
        rs = self.relu(self.conv2(rs))
        out = self.conv3(rs)
        out = out + lms

        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1500, gamma=0.5)
        return [optimizer], [scheduler]
    
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
    