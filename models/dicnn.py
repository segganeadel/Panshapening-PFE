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

from downsample import MTF
from torchmetrics.image.d_s import SpatialDistortionIndex
from torchmetrics.image.d_lambda import SpectralDistortionIndex
from torchmetrics.image.ergas import ErrorRelativeGlobalDimensionlessSynthesis
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.qnr import QualityWithNoReference


class DICNN(L.LightningModule):
    def __init__(self, spectral_num, channel=64, reg=True, satellite="qb", mtf_kernel_size=41, ratio=4):
        super(DICNN, self).__init__()
        self.spectral_num = spectral_num
        self.satellite = satellite
        self.ratio = ratio
        self.mtf_kernel_size = mtf_kernel_size

        self.reg = reg
        # ConvTranspose2d: output = (input - 1)*stride + outpading - 2*padding + kernelsize
        self.conv1 = nn.Conv2d(in_channels=spectral_num + 1,    out_channels=channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=channel,             out_channels=channel, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=channel,        out_channels=spectral_num, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.loss = nn.MSELoss()

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
    