import torch.utils.data as data
import torch
import h5py
import cv2
import numpy as np

##---------------------------------------------------------- Test dataset ----------------------------------------------------------##
    
class Dataset_h5py_test(data.Dataset):
    def __init__(self, file_path: str, img_scale: float = 2047.0, highpass = False, device = 'cpu'):
        super(Dataset_h5py_test, self).__init__()
        self.highpass = highpass
        self.img_scale = img_scale
        data = h5py.File(file_path)  # NxCxHxW = 0x1x2x3
        print(f"loading Dataset: {file_path} with {img_scale}")

        ms = data["ms"][...]
        self.ms = torch.as_tensor(ms, dtype=torch.float32) 

        lms = data["lms"][...] 
        self.lms = torch.as_tensor(lms, dtype=torch.float32)

        pan = data["pan"][...]
        self.pan = torch.as_tensor(pan, dtype=torch.float32)


    def __getitem__(self, index):

        lms = self.lms[index]
        lms /= self.img_scale
        
        pan = self.pan[index]
        pan =  pan if not self.highpass else pan - cv2.boxFilter(pan.numpy(), -1, (5, 5))
        pan /= self.img_scale

        ms = self.ms[index]
        ms = ms if not self.highpass else ms - cv2.boxFilter(ms.numpy(), -1, (5, 5))

        ms /= self.img_scale


        return [lms, pan, ms]
    
    def __len__(self):
        return self.pan.shape[0]
##---------------------------------------------------------- Train dataset ----------------------------------------------------------##

class Dataset_h5py_train(data.Dataset):
    def __init__(self, file_path: str, img_scale: float = 2047.0, highpass = False):
        super(Dataset_h5py_train, self).__init__()
        self.highpass = highpass
        self.img_scale = img_scale
        data = h5py.File(file_path,)  # NxCxHxW = 0x1x2x3
        print(f"loading Dataset: {file_path} with {img_scale}")
        self.img_scale = img_scale

        lms = data["lms"][...] 
        self.lms = torch.as_tensor(lms, dtype=torch.float32)
        
        pan = data["pan"][...]
        self.pan = torch.as_tensor(pan, dtype=torch.float32)
        
        ms = data["ms"][...]
        self.ms = torch.as_tensor(ms, dtype=torch.float32)
        
        gt = data['gt'][...]  # Nx1xHxW
        self.gt = torch.as_tensor(gt, dtype=torch.float32) # Nx1xHxW

    def __getitem__(self, index):

        lms = self.lms[index]
        lms /= self.img_scale
        
        pan = self.pan[index]
        pan =  pan if not self.highpass else pan - cv2.boxFilter(pan.numpy(), -1, (5, 5))
        pan /= self.img_scale

        ms = self.ms[index]
        ms = ms if not self.highpass else ms - cv2.boxFilter(ms.numpy(), -1, (5, 5))
        ms /= self.img_scale

        gt = self.gt[index]
        gt /= self.img_scale

        return [lms, pan, ms], gt
   
    def __len__(self):
        return self.pan.shape[0]

