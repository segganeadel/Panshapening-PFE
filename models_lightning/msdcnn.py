import torch
import torch.nn as nn
import lightning as L
from torchmetrics.image import SpectralAngleMapper, ErrorRelativeGlobalDimensionlessSynthesis

class MSDCNN(L.LightningModule):
    def __init__(self, spectral_num):
        super(MSDCNN, self).__init__()

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

        ##############################################################################################################
        # criterion
        self.criterion = nn.MSELoss()
        # metrics
        self.sam = SpectralAngleMapper()
        self.ergas = ErrorRelativeGlobalDimensionlessSynthesis(0.25)
        


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
        