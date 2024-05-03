from torch.utils.data import DataLoader
import lightning as L
import os

class PANDataModule(L.LightningDataModule):
    def __init__(self, data_dir, highpass, batch_size: int = 1):  
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        
        # Load data

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size)