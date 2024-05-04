import torch
import torch.nn as nn
import lightning as L
from torchmetrics.image import SpectralAngleMapper, ErrorRelativeGlobalDimensionlessSynthesis

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
    def __init__(self, spectral_num, channel=64):
        super(BDPN, self).__init__()

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
        
        #############################################################################################
        #criterion
        self.criterion = nn.MSELoss()
        #metrics
        self.sam = SpectralAngleMapper()
        self.ergas = ErrorRelativeGlobalDimensionlessSynthesis(0.25)


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

    def training_step(self, batch, batch_idx):
        y_hat = self(batch)

        y = batch['gt']
        loss = self.criterion(y_hat, y)   
        sam = self.sam(y_hat, y).rad2deg()
        ergas = self.ergas(y_hat, y)
        self.log_dict({'training_loss': loss, 
                       'training_sam': sam, 
                       'training_ergas': ergas}, 
                            on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        y_hat = self(batch)

        y = batch['gt']
        sam = self.sam(y_hat, y).rad2deg()
        ergas = self.ergas(y_hat, y)
        loss = self.criterion(y_hat, y)
        self.log_dict({'validation_loss': loss, 
                       'validation_sam': sam, 
                       'validation_ergas': ergas}, 
                            on_step=True, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        y_hat = self(batch)

        y = batch['gt']
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss)
        return loss

    def predict_step(self, batch, batch_idx):
        x = batch
        preds = self(x)
        return preds