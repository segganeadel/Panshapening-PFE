import os
import h5py
from scipy.io import savemat

data_path_in = os.path.join(".","data","h5py","qb","train","train_qb-001.h5")
data = h5py.File(data_path_in)

data_path_out = os.path.join(".","data_out")
os.makedirs(data_path_out, exist_ok=True)


ms = data.get("ms")
pan = data.get("pan")
gt = data.get("gt")
lms = data.get("lms")

for i in range(len(ms)):
    image_dict = {  
        "ms":ms[i],
        "pan":pan[i],
        "lms":lms[i],
        "gt":gt[i]
        }   
    savemat(os.path.join(data_path_out,f"{i}.mat"), image_dict)

