
import os
from models.apnn import APNN
from models.bdpn import BDPN
from models.dicnn import DICNN
from models.drpnn import DRPNN
from models.fusionnet import FusionNet
from models.msdcnn import MSDCNN
from models.pannet import PanNet
from models.pnn import PNN


import torch
import torch.nn as nn
import cv2 as cv

from dataloader import Dataset_h5py_test

models = {
    "APNN":(APNN,"apnn.pth"),
    "BDPN":(BDPN,"bdpn.pth"),
    "DICNN":(DICNN,"dicnn1.pth"),
    "DRPNN":(DRPNN,"drpnn.pth"),
    "FusionNet":(FusionNet,"fusionnet.pth"),
    "MSDCNN":(MSDCNN,"msdcnn.pth"),
    "PanNet":(PanNet,"panet.pth"),
    "PNN":(PNN,"pnn.pth")
    }

# model_name = "PanNet"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the data 
data_path1 = os.path.join(".","data","h5py","qb","full_examples","test_qb_OrigScale_multiExm1.h5")
data_path2 = os.path.join(".","data","h5py","qb","reduced_examples","test_qb_multiExm1.h5")

data = Dataset_h5py_test(data_path1, img_scale=2047.0, highpass=False, device=device)

data_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False, num_workers=0, drop_last=False)



def generate_image(model, ms, lms, pan, batch_n, model_name):

    result = model(ms, lms, pan)
    image_in = lms.detach().cpu().numpy()[:,:3].transpose(0,2,3,1)*255
    image_out = result.detach().cpu().numpy()[:,:3].transpose(0,2,3,1)*255

    for index,image in enumerate(image_in):
        print(image.shape, "image_in")
        print(image.max())
        string = f"out/in_images/image_in_{batch_n}_{index}.png"
        print(string)
        cv.imwrite(string, image)

    for index,image in enumerate(image_out):
        print(result.shape, "result")
        print(image.shape, "image_out")
        print(image.max())
        path_out = f"out/{model_name}/image_out_{batch_n}_{index}.png"
        cv.imwrite(path_out, image)



for model_name in models:

    model, wieght_path = models[model_name] # 4 spectral bands + 1 panchromatic band added inside the model
    model = model(4).to(device)
    path = os.path.join(".","weights","QB",wieght_path) 
    model.load_state_dict(torch.load(path))
    model.eval()

    iter_data = iter(data_loader)
    for index in range(len(data_loader)):
        ms, lms, pan = next(iter_data)
        generate_image(model, ms, lms, pan, index, model_name)
