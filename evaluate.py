import os
from scipy.io import loadmat
import numpy as np
import torch
from metrics.ERGAS import ERGAS
from metrics_torch.ERGAS_TORCH import ergas_torch

from metrics.SAM import SAM
from metrics_torch.SAM_TORCH import sam_torch

from metrics.q2n import q2n
from metrics_torch.Q2N_TORCH import q2n_torch

from metrics_torch.D_LAMBDA_K_TORCH import d_lambda_k_torch

mss = []
files_ms = sorted(os.listdir("./data/mat/qb/test/"), key=len)

for file in files_ms:
    ms = loadmat(os.path.join("./data/mat/qb/test/", file)).get("ms")
    mss.append(ms)

gts = []
files_gt = sorted(os.listdir("./data/mat/qb/test/"), key=len)

for file in files_gt:
    gt = loadmat(os.path.join("./data/mat/qb/test/", file)).get("gt")
    gts.append(gt)

outs = []
files_out = sorted(os.listdir("./out/"), key=len)

for file in files_out:
    out = loadmat(os.path.join("./out/", file)).get("out")
    outs.append(out)


q2n_indexes = []
ergas_indexes = []
sam_indexes = []

for images in zip(mss, gts,outs):
    
    ms_image, gt_image, out_image = images

    gt_image /= 2047
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

    D_lambda_K_index = d_lambda_k_torch(out_image_ten, ms_image_ten)
    print(D_lambda_K_index.item())


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