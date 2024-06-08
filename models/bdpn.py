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


# ----------------------------------------------------
class Resblock(nn.Module):
    def __init__(self):
        super(Resblock, self).__init__()

        channel = 64
        self.conv20 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1)
        self.conv21 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1)
        self.prelu = nn.PReLU(num_parameters = 1, init = 0.2)

    def forward(self, x):
        rs1 = self.prelu(self.conv20(x))  # Bsx32x64x64
        rs1 = self.conv21(rs1)  # Bsx32x64x64
        rs = torch.add(x, rs1)  # Bsx32x64x64

        return rs

# -----------------------------------------------------
class BDPN(L.LightningModule):
    def __init__(self, spectral_num, channel=64, satellite="qb", mtf_kernel_size=41, ratio=4):
        super(BDPN, self).__init__()
        self.satellite = satellite
        self.ratio = ratio
        self.mtf_kernel_size = mtf_kernel_size

        channel1 = channel
        self.spectral_num = spectral_num
        channel2 = 4*spectral_num

        # ConvTranspose2d: output = (input - 1)*stride + outpading - 2*padding + kernelsize
        # Conv2d: padding = kernel_size//2
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=channel1, kernel_size=3, stride=1, padding=1)
        #self.conv2 = nn.Conv2d(in_channels=channel1, out_channels=channel1, kernel_size=3, stride=1, padding=1,
        #                       bias=True)
        self.res1 = Resblock()
        self.res2 = Resblock()
        self.res3 = Resblock()
        self.res4 = Resblock()
        self.res5 = Resblock()
        self.res6 = Resblock()
        self.res7 = Resblock()
        self.res8 = Resblock()
        self.res9 = Resblock()
        self.res10 = Resblock()


        self.rres1 = Resblock()
        self.rres2 = Resblock()
        self.rres3 = Resblock()
        self.rres4 = Resblock()
        self.rres5 = Resblock()
        self.rres6 = Resblock()
        self.rres7 = Resblock()
        self.rres8 = Resblock()
        self.rres9 = Resblock()
        self.rres10 = Resblock()


        self.conv3 = nn.Conv2d(in_channels=channel1, out_channels=spectral_num, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=spectral_num, out_channels=channel2, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=spectral_num, out_channels=channel2, kernel_size=3, stride=1, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pixshuf = nn.PixelShuffle(upscale_factor=2)  # out = ps(img)
        self.prelu = nn.PReLU(num_parameters = 1, init = 0.2)


        self.backbone = nn.Sequential(  # method 2: 4 resnet repeated blocks
            self.res1,
            self.res2,
            self.res3,
            self.res4,
            self.res5,
            self.res6,
            self.res7,
            self.res8,
            self.res9,
            self.res10
        )

        self.backbone2 = nn.Sequential(  # method 2: 4 resnet repeated blocks
            self.rres1,
            self.rres2,
            self.rres3,
            self.rres4,
            self.rres5,
            self.rres6,
            self.rres7,
            self.rres8,
            self.rres9,
            self.rres10
        )

        self.loss = nn.MSELoss()
        
    def forward(self, input):  # x= ms(Nx8x16x16); y = pan(Nx1x64x64)
        
        ms = input["ms"]
        pan = input["pan"]
        
        x = ms
        y = pan

        # ========A): pan feature (extraction)===========
        # --------pan feature (stage 1:)------------
        pan_feature = self.conv1(y)  # Nx64x64x64
        rs = pan_feature  # Nx64x64x64

        rs = self.backbone(rs)  # Nx64x64x64

        pan_feature1 = torch.add(pan_feature, rs)  # Bsx64x64x64
        pan_feature_level1 = self.conv3(pan_feature1)  # Bsx8x64x64
        pan_feature1_out = self.maxpool(pan_feature1)  # Bsx64x32x32

        # --------pan feature (stage 2:)------------
        rs = pan_feature1_out  # Bsx64x32x32

        rs = self.backbone2(rs)  # Nx64x32x32, ????

        pan_feature2 = torch.add(pan_feature1_out, rs)  # Bsx64x32x32
        pan_feature_level2 = self.conv3(pan_feature2)  # Bsx8x32x32

        # ========B): ms feature (extraction)===========
        # --------ms feature (stage 1:)------------
        ms_feature1 = self.conv4(x)  # x= ms(Nx8x16x16); ms_feature1 =Nx32x16x16
        ms_feature_up1 = self.pixshuf(ms_feature1)  # Nx8x32x32
        ms_feature_level1 = torch.add(pan_feature_level2, ms_feature_up1)  # Nx8x32x32

        # --------ms feature (stage 2:)------------
        ms_feature2 = self.conv5(ms_feature_level1)  # Nx32x32x32
        ms_feature_up2 = self.pixshuf(ms_feature2)  # Nx8x64x64
        output = torch.add(pan_feature_level1, ms_feature_up2)  # Nx8x64x64

        return output
    
    def configure_optimizers(self):
        pass

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
    