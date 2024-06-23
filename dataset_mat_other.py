from torch.utils.data import Dataset
import cv2
from scipy.io import loadmat
import os
import numpy as np
from downsample import MTF, interp23tap_GPU


class Dataset_mat_rr(Dataset):
    def __init__(self, dir_path: str, sattelite: str, channels: int, img_scale: float = 2047.0, highpass = False,):
        super(Dataset_mat_rr, self).__init__()
        
        self.highpass = highpass
        self.img_scale = img_scale
        
        self.dir_path = dir_path

        self.ms_path = os.path.join(dir_path, 'MS_256')
        self.pan_path = os.path.join(dir_path, 'PAN_1024')

        self.files = sorted(os.listdir(self.ms_path), key=len)

        self.mtf = MTF(sattelite, channels, device='cpu')

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):

        file = self.files[index]

        hrms_mat = loadmat(os.path.join(self.ms_path, file))
        pan_mat = loadmat(os.path.join(self.pan_path, file))

        pan = pan_mat["imgPAN"]
        pan =  pan if not self.highpass else pan - cv2.boxFilter(pan, -1, (5, 5))
        pan = self.mtf.genMTF_pan_np(pan)

        gt = hrms_mat["imgMS"]

        hrms = hrms_mat["imgMS"]
        print(hrms.dtype, hrms.shape)
        ms = self.mtf.genMTF_ms_np(hrms)
        ms = ms if not self.highpass else ms - cv2.boxFilter(ms, -1, (5, 5))

        lms = interp23tap_GPU(ms, 4)

        pan = np.expand_dims(pan, 0)
        gt = gt.transpose(2, 0, 1)
        ms = ms.transpose(2, 0, 1)
        lms = lms.transpose(2, 0, 1)

        pan = pan / self.img_scale
        gt = gt / self.img_scale
        ms = ms / self.img_scale
        lms = lms / self.img_scale

        return {
            'ms':ms.astype(np.float32),
            'lms':lms.astype(np.float32),
            'pan':pan.astype(np.float32),
            'gt':gt.astype(np.float32)
            }

   
class Dataset_mat_fr(Dataset):
    def __init__(self, dir_path: str, sattelite: str, channels: int, img_scale: float = 2047.0, highpass = False,):
        super(Dataset_mat_fr, self).__init__()
        
        self.highpass = highpass
        self.img_scale = img_scale
        
        self.dir_path = dir_path

        self.ms_path = os.path.join(dir_path, 'MS_256')
        self.pan_path = os.path.join(dir_path, 'PAN_1024')

        self.files = sorted(os.listdir(self.ms_path), key=len)

        self.mtf = MTF(sattelite, channels, device='cpu')

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):

        file = self.files[index]

        hrms_mat = loadmat(os.path.join(self.ms_path, file))
        pan_mat = loadmat(os.path.join(self.pan_path, file))

        pan = pan_mat["imgPAN"]
        pan =  pan if not self.highpass else pan - cv2.boxFilter(pan, -1, (5, 5))

        ms = hrms_mat["imgMS"]
        ms = ms if not self.highpass else ms - cv2.boxFilter(ms, -1, (5, 5))

        lms = interp23tap_GPU(hrms_mat["imgMS"], 4)

        pan = np.expand_dims(pan, 0)
        ms = ms.transpose(2, 0, 1)
        lms = lms.transpose(2, 0, 1)

        pan = pan / self.img_scale
        ms = ms / self.img_scale
        lms = lms / self.img_scale

        return {
            'ms':ms.astype(np.float32),
            'lms':lms.astype(np.float32),
            'pan':pan.astype(np.float32),
            }

   
