from torch.utils.data import Dataset
import cv2
from scipy.io import loadmat
import os
import numpy as np


class Dataset_mat_rr(Dataset):
    def __init__(self, dir_path: str, img_scale: float = 2047.0, highpass = False):
        super(Dataset_mat_rr, self).__init__()
        
        self.highpass = highpass
        self.img_scale = img_scale
        self.img_scale = img_scale
        
        self.dir_path = dir_path
        self.files = sorted(os.listdir(dir_path), key=len)

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):

        file = self.files[index]
        data = loadmat(os.path.join(self.dir_path, file))


        lms = data["lms"]
        lms /= self.img_scale
        
        pan = data["pan"]
        pan =  pan if not self.highpass else pan - cv2.boxFilter(pan, -1, (5, 5))
        pan /= self.img_scale

        ms = data["ms"]
        ms = ms if not self.highpass else ms - cv2.boxFilter(ms, -1, (5, 5))
        ms /= self.img_scale

        gt = data['gt']  
        gt /= self.img_scale

        return {
            'ms':ms.astype(np.float32),
            'lms':lms.astype(np.float32),
            'pan':pan.astype(np.float32),
            'gt':gt.astype(np.float32)
            }

   
class Dataset_mat_fr(Dataset):
    def __init__(self, dir_path: str, img_scale: float = 2047.0, highpass = False):
        super(Dataset_mat_fr, self).__init__()
        
        self.highpass = highpass
        self.img_scale = img_scale
        self.img_scale = img_scale
        
        self.dir_path = dir_path
        self.files = sorted(os.listdir(dir_path), key=len)

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):

        file = self.files[index]
        data = loadmat(os.path.join(self.dir_path, file))


        lms = data["lms"]
        lms /= self.img_scale
        
        pan = data["pan"]
        pan =  pan if not self.highpass else pan - cv2.boxFilter(pan, -1, (5, 5))
        pan /= self.img_scale

        ms = data["ms"]
        ms = ms if not self.highpass else ms - cv2.boxFilter(ms, -1, (5, 5))
        ms /= self.img_scale

        print("dataloader shape", pan.shape, ms.shape, lms.shape)
        return {
            'ms':ms.astype(np.float32),
            'lms':lms.astype(np.float32),
            'pan':pan.astype(np.float32),
            }

   
