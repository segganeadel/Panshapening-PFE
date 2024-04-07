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
        data = h5py.File(file_path)  # NxCxHxW = 0x1x2x3
        print(f"loading Dataset: {file_path} with {img_scale}")


        ms = data["ms"][...]
        self.ms = torch.as_tensor(ms, dtype=torch.float32) 
        if self.highpass : self.ms -= cv2.boxFilter(self.ms.numpy(), -1, (5, 5))
        self.ms /= img_scale
        self.ms = self.ms.to(device)

        lms = data["lms"][...] 
        self.lms = torch.as_tensor(lms, dtype=torch.float32)
        self.lms /= img_scale
        self.lms = self.lms.to(device)

        pan = data["pan"][...]
        self.pan = torch.as_tensor(pan, dtype=torch.float32)
        if self.highpass : self.pan -= cv2.boxFilter(self.pan.numpy(), -1, (5, 5))
        self.pan /= img_scale
        self.pan = self.pan.to(device)

    def __getitem__(self, index):

        ms = self.ms[index] 
        lms = self.lms[index]
        pan = self.pan[index] 
        return ms, lms, pan
    
    def __len__(self):
        return self.pan.shape[0]
##---------------------------------------------------------- Train dataset ----------------------------------------------------------##

class Dataset_h5py_train(data.Dataset):
    def __init__(self, file_path: str, img_scale: float = 2047.0, highpass = False):
        super(Dataset_h5py_train, self).__init__()
        self.highpass = highpass
        data = h5py.File(file_path,)  # NxCxHxW = 0x1x2x3
        print(f"loading Dataset: {file_path} with {img_scale}")
        self.img_scale = img_scale

        pan = data["pan"][...]
        self.pan = torch.as_tensor(pan, dtype=torch.float32)
        lms = data["lms"][...] 
        self.lms = torch.as_tensor(lms, dtype=torch.float32)
        ms = data["ms"][...]
        self.ms = torch.as_tensor(ms, dtype=torch.float32)
        gt = data['gt'][...]  # Nx1xHxW
        self.gt = torch.as_tensor(gt, dtype=torch.float32) # Nx1xHxW

    def __getitem__(self, index):

        ms = self.ms[index] if not self.highpass else self.ms[index] - cv2.boxFilter(self.ms[index].numpy(), -1, (5, 5))
        ms = self.ms[index] / self.img_scale

        lms = self.lms[index] / self.img_scale
        
        pan = self.pan[index] if not self.highpass else self.pan[index] - cv2.boxFilter(self.pan[index].numpy(), -1, (5, 5))
        pan = pan / self.img_scale

        gt = self.gt[index] / self.img_scale

        return ms, lms, pan, gt
   
    def __len__(self):
        return self.pan.shape[0]

