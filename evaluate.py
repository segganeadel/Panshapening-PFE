import h5py
import numpy as np
import torch

from metrics.MTF import MTF
from metrics_numpy.MTF import MTF as new_MTF

data = h5py.File("./mytestfile.hdf5", "r")

gt = data.get("gt_data")
out = data.get("out_data")
gt = np.array(gt)
out = np.array(out)

metrics = []
for images in zip(gt,out):
    gt_images,out_images = images
    
    # gt_images /= 2047
    # out_images /= 2047.0

    gt_image = gt_images[0]
    out_image = out_images[0]

    gt_image_ten = torch.as_tensor(gt_images)
    out_image_ten =torch.as_tensor(out_images)

    gt_image_t = np.copy(gt_image).transpose(1,2,0)
    out_image_t = np.copy(out_image).transpose(1,2,0)


    new_d = new_MTF(np.copy(gt_image_t),"QB", 4)
    d =         MTF(gt_image_t, "QB", 4)

    print(new_d.shape, d.shape)
    err = np.mean(d - new_d)
    print(err)
#     metric = [SAM_index,ERGAS_index.item()]

#     print(metric)
#     metrics.append(metric)

# with open('fusionnet_rr_after.csv', 'w',newline='') as file:
#     writer = csv.writer(file)
#     field = ["SAM", "ERGAS"]
#     writer.writerow(field)
#     writer.writerows(metrics)