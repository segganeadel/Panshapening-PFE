import os
from scipy.io import loadmat
import numpy as np
import torch
import cv2

from metrics.ERGAS import ERGAS
from metrics_torch.ERGAS_TORCH import ergas_torch

from metrics.SAM import SAM
from metrics_torch.SAM_TORCH import sam_torch

from metrics.q2n import q2n
from metrics_torch.Q2N_TORCH import q2n_torch

from downsample import MTF, interp23tap_GPU


files_in = sorted(os.listdir("./data/mat/qb/test/"), key=len)

mss = []
for file in files_in:
    ms = loadmat(os.path.join("./data/mat/qb/test/", file)).get("ms")
    mss.append(ms)

gts = []
for file in files_in:
    gt = loadmat(os.path.join("./data/mat/qb/test/", file)).get("gt")
    gts.append(gt)

pans = []
for file in files_in:
    pan = loadmat(os.path.join("./data/mat/qb/test/", file)).get("pan")
    pans.append(pan)

lmss = []
for file in files_in:
    lms = loadmat(os.path.join("./data/mat/qb/test/", file)).get("lms")
    lmss.append(lms)


files_out = sorted(os.listdir("./out/"), key=len)
outs = []
for file in files_out:
    out = loadmat(os.path.join("./out/", file)).get("out")
    outs.append(out)


q2n_indexes = []
ergas_indexes = []
sam_indexes = []

pans= np.stack(pans)
gts = np.stack(gts)

mtf = MTF(sensor= "qb", channels= 4,ratio= 4,kernel_size= 41, device="cpu")
print(pans.shape)
# pan_lr = mtf.genMTF_pan_np((pans[0]/2047.0).astype(np.float32).squeeze())
pan_lr = mtf.genMTF_pan_torch(torch.as_tensor(pans, dtype=torch.float32)/2047.0)
print(pan_lr.shape)
cv2.imwrite("pan_lr.png", pan_lr.numpy().transpose(0, 2, 3, 1)[0]*255)

# ms_lr = mtf.genMTF_ms_np((gts[0]/2047.0).astype(np.float32).transpose(1,2,0))
ms_lr = mtf.genMTF_ms_torch(torch.as_tensor(gts, dtype=torch.float32)/2047.0)
print(ms_lr.shape)
cv2.imwrite("ms_lr.png", ms_lr.numpy().transpose(0, 2, 3, 1)[0,:,:,:3]*255)

ms_lr = ms_lr[0].numpy().transpose(1,2,0)
lms = interp23tap_GPU(ms_lr, 4)[:,:,:3]
print(lms.shape)
cv2.imwrite("lms.png", lms*255)


for images in zip(mss, gts,outs):
    
    ms_image, gt_image, out_image = images

    gt_image /= 2047.0
    out_image /= 2047.0

    ms_image_n = np.copy(ms_image).transpose(1,2,0)
    gt_image_n = np.copy(gt_image).transpose(1,2,0)
    out_image_n = np.copy(out_image).transpose(1,2,0)

    
    ms_image_ex = np.expand_dims(ms_image, axis=0)
    gt_image_ex = np.expand_dims(gt_image, axis=0)
    out_image_ex = np.expand_dims(out_image, axis=0)
    
    ms_image_ten = torch.as_tensor(ms_image_ex).to('cuda')
    gt_image_ten = torch.as_tensor(gt_image_ex).to('cuda')
    out_image_ten =torch.as_tensor(out_image_ex).to('cuda')




    # Q2N_index, _ = q2n(gt_image_n, out_image_n, Q_blocks_size=32, Q_shift=32)
    # Q2N_index_torch = q2n_torch(out_image_ten, gt_image_ten)
    # print(Q2N_index, Q2N_index_torch.item())
    # print("err", Q2N_index - Q2N_index_torch.item())
    # q2n_indexes.append(Q2N_index)


    # SAM_index, _ = SAM(gt_image_n, out_image_n)
    # SAM_index_torch = sam_torch(out_image_ten, gt_image_ten)
    # print(SAM_index, SAM_index_torch.item())
    # print("err", SAM_index - SAM_index_torch.item())
    # sam_indexes.append(SAM_index)

    # ERGAS_index_np = ERGAS(gt_image_n, out_image_n,ratio=4)   
    # ERGAS_index_torch = ergas_torch(out_image_ten, gt_image_ten)    
    # print(ERGAS_index_np, ERGAS_index_torch.item())
    # print("err", ERGAS_index_np - ERGAS_index_torch.item())
    # ergas_indexes.append(ERGAS_index_np)

# outs_s = torch.as_tensor(np.stack(outs)).to('cuda')
# gts_s = torch.as_tensor(np.stack(gts)).to('cuda')

# q2n_index_mean = np.stack(sam_indexes).mean()
# Q2N_index_torch_mean = sam_torch(outs_s, gts_s)

# print (q2n_index_mean, Q2N_index_torch_mean.item())
# print("err batch", q2n_index_mean - Q2N_index_torch_mean.item())