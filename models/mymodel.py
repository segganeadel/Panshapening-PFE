
import torch
import torch
import torch.nn as nn
import lightning as L
import torch.nn.functional as F
from metrics_torch.ERGAS_TORCH import ergas_torch
from metrics_torch.SAM_TORCH import sam_torch
from .mamba_helper.vssm import VSSM

class MambFuse(L.LightningModule):
    def __init__(self, spectral_num, channel=32):
        super(MambFuse, self).__init__()
        self.vssm = VSSM(
            patch_size= 4,
            in_chans= spectral_num,
            depths= [2,2,2,2],
        )

    def forward(self, input):
        lms = input['lms']
        pan = input['pan']

        pan_concat = pan.repeat(1, self.spectral_num, 1, 1)  # Bsx8x64x64
        diff = torch.sub(pan_concat, lms)
        
        out = self.vssm(diff)
        print ("out shape", out.shape)
        return out

            

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=5e-4)
    
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
    