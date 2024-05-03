from torch.utils.data import Dataset
import numpy as np
import h5py
import cv2

##---------------------------------------------------------- RR dataset ----------------------------------------------------------##

class Dataset_h5py_rr(Dataset):
    def __init__(self, file_path: str, img_scale: float = 2047.0, highpass = False):
        super(Dataset_h5py_rr, self).__init__()
        
        self.highpass = highpass
        self.img_scale = img_scale
        data = h5py.File(file_path)  # NxCxHxW = 0x1x2x3
        print(f"loading Dataset: {file_path} with {img_scale}")
        self.img_scale = img_scale

        lms = data["lms"]
        self.lms = np.asarray(lms)
        
        pan = data["pan"]
        self.pan = np.asarray(pan)
        
        ms = data["ms"]
        self.ms = np.asarray(ms)
        
        gt = data['gt']  # Nx1xHxW
        self.gt = np.asarray(gt) # Nx1xHxW

    def __getitem__(self, index):

        lms = self.lms[index]
        lms /= self.img_scale
        
        pan = self.pan[index]
        pan =  pan if not self.highpass else pan - cv2.boxFilter(pan, -1, (5, 5))
        pan /= self.img_scale

        ms = self.ms[index]
        ms = ms if not self.highpass else ms - cv2.boxFilter(ms, -1, (5, 5))
        ms /= self.img_scale

        gt = self.gt[index]
        gt /= self.img_scale

        return {
            'ms':ms.astype(np.float32),
            'lms':lms.astype(np.float32),
            'pan':pan.astype(np.float32),
            'gt':gt.astype(np.float32)
            }

   
    def __len__(self):
        return self.pan.shape[0]
##---------------------------------------------------------- FR dataset ----------------------------------------------------------##
    
class Dataset_h5py_fr(Dataset):
    def __init__(self, file_path: str, img_scale: float = 2047.0, highpass = False):
        super(Dataset_h5py_fr, self).__init__()
        
        self.highpass = highpass
        self.img_scale = img_scale
        data = h5py.File(file_path)  # NxCxHxW = 0x1x2x3
        print(f"loading Dataset: {file_path} with {img_scale}")

        lms = data["lms"] 
        self.lms = np.asarray(lms)
        
        pan = data["pan"]
        self.pan = np.asarray(pan)
        
        ms = data["ms"]
        self.ms = np.asarray(ms)

    def __getitem__(self, index):

        lms = self.lms[index]
        lms /= self.img_scale
        
        pan = self.pan[index]
        pan =  pan if not self.highpass else pan - cv2.boxFilter(pan, -1, (5, 5))
        pan /= self.img_scale

        ms = self.ms[index]
        ms = ms if not self.highpass else ms - cv2.boxFilter(ms, -1, (5, 5))

        ms /= self.img_scale


        return {
            'ms':ms.astype(np.float32),
            'lms':lms.astype(np.float32),
            'pan':pan.astype(np.float32)
            }
    def __len__(self):
        return self.pan.shape[0]

