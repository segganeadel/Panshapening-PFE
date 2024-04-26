import h5py
import numpy as np
import torch

from metrics_torch.ERGAS_TORCH import ergas_torch
from metrics_torch.SAM_TORCH import sam_torch
from metrics_numpy.SAM import SAM as new_SAM

from metrics.q2n import q2n
from metrics_torch.Q2N_TORCH import q2n_torch

data = h5py.File("./mytestfile.hdf5", "r")

gt = data.get("gt_data")
out = data.get("out_data")
gt = np.array(gt)
out = np.array(out)

metrics = []
for images in zip(gt,out):
    gt_images,out_images = images

    gt_image = gt_images[0]
    out_image = out_images[0]

    gt_image_t = np.copy(gt_image).transpose(1,2,0)
    out_image_t = np.copy(out_image).transpose(1,2,0)

    gt_image_ten = torch.as_tensor(gt_images)
    out_image_ten =torch.as_tensor(out_images)

    # ERGAS_index = ERGAS(gt_image,out_image)
    # SAM_index,SAM_map = SAM(gt_image,out_image)

    q2n_index, _ = q2n(gt_image_t, out_image_t,32,32)
    new_q2n_index, _ = q2n_torch(gt_image_ten, out_image_ten)
    new_q2n_index = new_q2n_index.item()

    print(new_q2n_index,q2n_index)
    if q2n_index != new_q2n_index:
        print("err",q2n_index,new_q2n_index)
        print("err",q2n_index-new_q2n_index)
    

#     metric = [SAM_index,ERGAS_index.item()]

#     print(metric)
#     metrics.append(metric)

# with open('fusionnet_rr_after.csv', 'w',newline='') as file:
#     writer = csv.writer(file)
#     field = ["SAM", "ERGAS"]
#     writer.writerow(field)
#     writer.writerows(metrics)