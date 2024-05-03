from argparse import ArgumentParser

import torch

from models_lightning.apnn import APNN
from models_lightning.bdpn import BDPN
from models_lightning.dicnn import DICNN
from models_lightning.drpnn import DRPNN
from models_lightning.fusionnet import FusionNet
from models_lightning.msdcnn import MSDCNN
from models_lightning.pannet import PanNet
from models_lightning.pnn import PNN
import os

from lightning import Trainer

from torch.utils.data import DataLoader
from dataloader import Dataset_h5py_fr, Dataset_h5py_rr
import cv2 as cv


def main(hparams):
    
    models = {
        "apnn":     (APNN,      "apnn.pth",     False),
        "bdpn":     (BDPN,      "bdpn.pth",     False),
        "dicnn":    (DICNN,     "dicnn1.pth",   False),
        "drpnn":    (DRPNN,     "drpnn.pth",    False),
        "fusionnet":(FusionNet, "fusionnet.pth",False),
        "msdcnn":   (MSDCNN,    "msdcnn.pth",   False),
        "pannet":   (PanNet,    "panet.pth",    True),
        "pnn":      (PNN,       "pnn.pth",      False)
    }

    # Choose the model
    model_name = "fusionnet"
    model, weights_path, highpass = models.get(model_name)
    weights_path = os.path.join(".","weights","QB", weights_path) 
    
    model = model(4)
    model.load_state_dict(torch.load(weights_path))

    # data_path1 = os.path.join(".","data","h5py","qb","full_examples","test_qb_OrigScale_multiExm1.h5")
    # data = Dataset_h5py_fr(data_path1, img_scale=2047.0, highpass=highpass)

    data_path2 = os.path.join(".","data","h5py","qb","reduced_examples","test_qb_multiExm1.h5")
    data = Dataset_h5py_rr(data_path2, img_scale=2047.0, highpass=highpass)
    
    dataloader = DataLoader(data, shuffle=False)
    

    trainer = Trainer()
    trainer.validate(model, dataloader)




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default=None)
    parser.add_argument("--devices", default=None)
    args = parser.parse_args()

    main(args)