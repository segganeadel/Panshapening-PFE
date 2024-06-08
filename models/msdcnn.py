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



class MSDCNN(L.LightningModule):
    def __init__(self, spectral_num, satellite="qb", mtf_kernel_size=41, ratio=4):
        super(MSDCNN, self).__init__()
        self.spectral_num = spectral_num
        self.satellite = satellite
        self.ratio = ratio
        self.mtf_kernel_size = mtf_kernel_size

        input_channel = spectral_num + 1
        output_channel = spectral_num

        self.conv1 =    nn.Conv2d(in_channels=input_channel,    out_channels=60, kernel_size=7, stride=1, padding=3)

        self.conv2_1 =  nn.Conv2d(in_channels=60,               out_channels=20, kernel_size=3, stride=1, padding=1)
        self.conv2_2 =  nn.Conv2d(in_channels=60,               out_channels=20, kernel_size=5, stride=1, padding=2)
        self.conv2_3 =  nn.Conv2d(in_channels=60,               out_channels=20, kernel_size=7, stride=1, padding=3)

        self.conv3 =    nn.Conv2d(in_channels=60,               out_channels=30, kernel_size=3, stride=1, padding=1)

        self.conv4_1 =  nn.Conv2d(in_channels=30,               out_channels=10, kernel_size=3, stride=1, padding=1)
        self.conv4_2 =  nn.Conv2d(in_channels=30,               out_channels=10, kernel_size=5, stride=1, padding=2)
        self.conv4_3 =  nn.Conv2d(in_channels=30,               out_channels=10, kernel_size=7, stride=1, padding=3)

        self.conv5 =    nn.Conv2d(in_channels=30,   out_channels=output_channel, kernel_size=5, stride=1, padding=2)

        self.shallow1 = nn.Conv2d(in_channels=input_channel,    out_channels=64, kernel_size=9, stride=1, padding=4)
        self.shallow2 = nn.Conv2d(in_channels=64,               out_channels=32, kernel_size=1, stride=1, padding=0)
        self.shallow3 = nn.Conv2d(in_channels=32,   out_channels=output_channel, kernel_size=5, stride=1, padding=2)

        self.relu = nn.ReLU(inplace=True)

        self.loss = nn.MSELoss()

    def forward(self, input):

        lms = input["lms"]
        pan = input["pan"]
    
        concat = torch.cat([lms, pan], 1)  # Bsx9x64x64

        out1 = self.relu(self.conv1(concat))  # Bsx60x64x64
        out21 = self.conv2_1(out1)   # Bsx20x64x64
        out22 = self.conv2_2(out1)   # Bsx20x64x64
        out23 = self.conv2_3(out1)   # Bsx20x64x64
        out2 = torch.cat([out21, out22, out23], 1)  # Bsx60x64x64

        out2 = self.relu(torch.add(out2, out1))  # Bsx60x64x64

        out3 = self.relu(self.conv3(out2))  # Bsx30x64x64
        out41 = self.conv4_1(out3)          # Bsx10x64x64
        out42 = self.conv4_2(out3)          # Bsx10x64x64
        out43 = self.conv4_3(out3)          # Bsx10x64x64
        out4 = torch.cat([out41, out42, out43], 1)  # Bsx30x64x64

        out4 = self.relu(torch.add(out4, out3))  # Bsx30x64x64

        out5 = self.conv5(out4)  # Bsx8x64x64

        shallow1 = self.relu(self.shallow1(concat))   # Bsx64x64x64
        shallow2 = self.relu(self.shallow2(shallow1))  # Bsx32x64x64
        shallow3 = self.shallow3(shallow2) # Bsx8x64x64

        out = torch.add(out5, shallow3)  # Bsx8x64x64
        out = self.relu(out)  # Bsx8x64x64

        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
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
    