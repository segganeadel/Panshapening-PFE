
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from metrics_torch.ERGAS_TORCH import ergas_torch
from metrics_torch.Q2N_TORCH import q2n_torch
from metrics_torch.SAM_TORCH import sam_torch

from downsample import MTF
from torchmetrics.image.d_s import SpatialDistortionIndex
from torchmetrics.image.d_lambda import SpectralDistortionIndex
from torchmetrics.image.ergas import ErrorRelativeGlobalDimensionlessSynthesis
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.qnr import QualityWithNoReference


from .mamba_helper.mamba import deepFuse
try:
    import lightning as L
except:
    import pytorch_lightning as L

class Resblock(nn.Module):
    def __init__(self):
        super(Resblock, self).__init__()

        channel = 32
        self.conv20 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1)
        self.conv21 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # x= hp of ms; y = hp of pan
        rs1 = self.relu(self.conv20(x))  # Bsx32x64x64
        rs1 = self.conv21(rs1)  # Bsx32x64x64
        rs = torch.add(x, rs1)  # Bsx32x64x64
        return rs

class MambFuse(L.LightningModule):
    def __init__(self, spectral_num, channel=32, satellite="qb", mtf_kernel_size=41, ratio=4):
        super(MambFuse, self).__init__()
        self.spectral_num = spectral_num
        self.satellite = satellite
        self.ratio = ratio
        self.mtf_kernel_size = mtf_kernel_size

        self.backbone_recept = nn.Sequential(  # method 2: 4 resnet repeated blocks
            nn.Conv2d(in_channels=spectral_num, out_channels=channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            Resblock(),
            Resblock(),
            Resblock(),
            Resblock(),
            nn.Conv2d(in_channels=channel, out_channels=spectral_num, kernel_size=3, stride=1, padding=1)
        )
        self.deepfusion = deepFuse(device=self.device, spectral_num=spectral_num)

        ############################################################################################################
        # Loss
        self.loss = nn.L1Loss()



    def forward(self, input):
        lms = input['lms']
        pan = input['pan']

        pan_concat = pan.repeat(1, self.spectral_num, 1, 1)  # Bsx8x64x64
        out = torch.sub(pan_concat, lms)

        out = self.backbone_recept(out)
        out = self.deepfusion(out)

        output = torch.add(out, lms) 
        return output

    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=2e-4)
    
    def setup(self, stage):
        if stage == 'test':
            ############################################################################################################
            # MTF
            self.mtf = MTF(sensor=self.satellite, 
                    channels= self.spectral_num,
                    device=self.device,
                    ratio=self.ratio,
                    kernel_size=self.mtf_kernel_size
                    )
            ############################################################################################################
            # Metrics 
            self.spatial_distortion_index_test = SpatialDistortionIndex()
            self.spectral_distortion_index_test = SpectralDistortionIndex()
            self.ergas_test = ErrorRelativeGlobalDimensionlessSynthesis()
            self.ssim_test = StructuralSimilarityIndexMeasure()
            self.psnr_test = PeakSignalNoiseRatio((0,1))
            self.qnr_test = QualityWithNoReference()

    def training_step(self, batch, batch_idx):
        y_hat = self(batch)

        y = batch['gt']
        loss = self.loss(y_hat, y)
        with torch.no_grad():
            ergas = ergas_torch(y_hat, y) 
            sam = sam_torch(y_hat, y)
            self.log_dict({'training_loss': loss, 
                        'training_sam':   sam, 
                        'training_ergas': ergas}, 
                            prog_bar=True,
                            sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        y_hat = self(batch)

        y = batch['gt']
        loss = self.loss(y_hat, y)
        
        with torch.no_grad():
            ergas = ergas_torch(y_hat, y)  
            sam = sam_torch(y_hat, y)
            self.log_dict({'validation_loss':  loss, 
                        'validation_sam':   sam, 
                        'validation_ergas': ergas}, 
                            prog_bar=True,
                            sync_dist=True)
        return loss
    
    def test_step(self, batch:dict, batch_idx):
        with torch.no_grad():
            y_hat = self(batch)
            y = batch.get('gt')
            
            # Reduced resolution mode
            if y is not None:
                self.ergas_test.update(y_hat, y)
                self.ssim_test.update(y_hat, y)
                self.psnr_test.update(y_hat, y)
                sam = sam_torch(y_hat, y)
                q2n = q2n_torch(y_hat, y)       
                self.log_dict({#'test_loss':  loss, 
                            'test_ergas': self.ergas_test,
                            'test_sam':  sam, 
                            'test_q2n': q2n,
                            'test_ssim': self.ssim_test,
                            'test_psnr': self.psnr_test}, 
                                prog_bar=True)
            # Full resolution mode
            else:
                pans = batch["pan"].repeat(1, self.spectral_num, 1, 1)
                down_pan = self.mtf.genMTF_pan_torch(batch["pan"])
                down_pans= down_pan.repeat(1, self.spectral_num, 1, 1)

                self.spatial_distortion_index_test.update(y_hat, {"ms":batch["ms"],"pan": pans,"pan_lr": down_pans})
                self.spectral_distortion_index_test.update(y_hat, batch['ms'])
                self.qnr_test.update(y_hat, {"ms":batch["ms"],"pan": pans,"pan_lr": down_pans})
                self.log_dict({"test_spatial_distortion": self.spatial_distortion_index_test,
                               "test_spectral_distortion": self.spectral_distortion_index_test,
                               "test_qnr": self.qnr_test}, 
                                prog_bar=True)


    def predict_step(self, batch, batch_idx):
        x = batch
        preds = self(x)
        return preds
    