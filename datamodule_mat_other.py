import os
from torch.utils.data import DataLoader
from dataset_mat_other import Dataset_mat_rr, Dataset_mat_fr
try:
    import lightning as L
except:
    import pytorch_lightning as L


class PANDataModule(L.LightningDataModule):
    def __init__(self,
                 data_dir, 
                 img_scale, 
                 highpass, 
                 num_workers = 0 , 
                 shuffle_train = False, 
                 batch_size: int = 1):  
        super().__init__()

        self.data_dir = data_dir
        self.img_scale = img_scale
        self.highpass = highpass
        self.num_workers = num_workers
        self.shuffle_train = shuffle_train
        self.batch_size = batch_size

        # Load data

    def train_dataloader(self):
        # getting the path of the first file in the train directory
        train_file_path = os.path.join(self.data_dir, 'train')

        # creating the dataset
        data = Dataset_mat_rr(train_file_path, img_scale=self.img_scale, highpass=self.highpass)
        return DataLoader(data, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, shuffle=self.shuffle_train, pin_memory=True)

    def val_dataloader(self):
        val_file_path = os.path.join(self.data_dir, 'valid')

        data = Dataset_mat_rr(val_file_path, img_scale=self.img_scale, highpass=self.highpass)
        return DataLoader(data, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        test_file_path = os.path.join(self.data_dir, 'test')

        data = Dataset_mat_rr(test_file_path, img_scale=self.img_scale, highpass=self.highpass)
        return DataLoader(data, batch_size=self.batch_size, num_workers=self.num_workers)    
    
    def predict_dataloader(self):
        predict_file_path = os.path.join(self.data_dir, 'predict')

        data = Dataset_mat_fr(predict_file_path, img_scale=self.img_scale, highpass=self.highpass)
        return DataLoader(data, batch_size=self.batch_size, num_workers=self.num_workers)
