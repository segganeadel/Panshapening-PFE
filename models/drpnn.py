import torch
import torch.nn as nn
import torch.nn.functional as F
from metrics_torch.ERGAS_TORCH import ergas_torch
from metrics_torch.Q2N_TORCH import q2n_torch
from metrics_torch.SAM_TORCH import sam_torch
try:
    import lightning as L
except:
    import pytorch_lightning as L



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

class DRPNN(L.LightningModule):
    def __init__(self, spectral_num, channel=32):
        super(DRPNN, self).__init__()

        # ConvTranspose2d: output = (input - 1)*stride + outpading - 2*padding + kernelsize
        self.conv1 = nn.Conv2d(in_channels=spectral_num+1,  out_channels=channel,           kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(in_channels=channel,         out_channels=spectral_num+1,    kernel_size=7, stride=1, padding=3)
        self.conv3 = nn.Conv2d(in_channels=spectral_num+1,  out_channels=spectral_num,      kernel_size=7, stride=1, padding=3)
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

    def forward(self, input):  # x= lms; y = pan
        lms = input["lms"]
        pan = input["pan"]
        
        input = torch.cat([lms, pan], 1)  # Bsx9x64x64
        rs = self.relu(self.conv1(input))  # Bsx64x64x64

        rs = self.backbone(rs)  # backbone!  Bsx64x64x64

        out_res = self.conv2(rs)  # Bsx9x64x64
        output1 = torch.add(input, out_res)  # Bsx9x64x64
        output  = self.conv3(output1)  # Bsx8x64x64

        return output
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
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

        with torch.no_grad():
            sam = sam_torch(y_hat, y)
            ergas = ergas_torch(y_hat, y)
            q2n = q2n_torch(y_hat, y)
            
            self.log_dict({#'test_loss':  loss, 
                        'test_sam':   sam, 
                        'test_ergas': ergas,
                        'test_q2n': q2n}, 
                            prog_bar=True)

    def predict_step(self, batch, batch_idx):
        x = batch
        preds = self(x)
        return preds
    