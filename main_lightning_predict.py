import os
from argparse import ArgumentParser

from models.apnn import APNN
from models.bdpn import BDPN
from models.dicnn import DICNN
from models.drpnn import DRPNN
from models.fusionnet import FusionNet
from models.msdcnn import MSDCNN
from models.pannet import PanNet
from models.pnn import PNN
from models.mambfuse import MambFuse

import torch
from datamodule_mat import PANDataModule
try:
    from lightning import Trainer
    from lightning.pytorch.loggers import WandbLogger, CSVLogger
except:
    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import WandbLogger, CSVLogger

import cv2 as cv
import numpy as np

from visualize import linstretch


def main(hparams):
    
    models = {
    # "model_name": (model,     weights_path, highpass)
        "apnn":     (APNN,      "apnn.pth",     False),
        "bdpn":     (BDPN,      "bdpn.pth",     False),
        "dicnn":    (DICNN,     "dicnn1.pth",   False),
        "drpnn":    (DRPNN,     "drpnn.pth",    False),
        "fusionnet":(FusionNet, "fusionnet.pth",False),
        "msdcnn":   (MSDCNN,    "msdcnn.pth",   False),
        "pannet":   (PanNet,    "pannet.pth",    True),
        "pnn":      (PNN,       "pnn.pth",      False),
        "mambfuse": (MambFuse,  "mambfuse.ckpt",False)
    }

    model_name = hparams.method
    satelite = hparams.satellite # mtf support ["qb", "ikonos", "geoeye1", "wv2", "wv3", "wv4"]
    data_dir = hparams.data_dir
    img_scale = hparams.img_scale

    model, weights_path, highpass = models.get(model_name)
    weights_path = os.path.join(".", "weights", satelite, weights_path)


    wandb_logger = WandbLogger(name=model_name, project="PanSharpening", prefix=satelite)
    csv_logger = CSVLogger(".")
    trainer = Trainer(logger=[wandb_logger, csv_logger], devices=1, num_nodes=1)
    
    channels_dict = {
        "qb": (4, np.index_exp[:,0:3]),
        # "ikonos": 4,
        # "geoeye1": 4,
        "wv2": (8,np.index_exp[:,(1,2,4)]),
        "wv3": (8, np.index_exp[:,(1,2,4)]),
        "wv4": 4
    }
    num_channels, slice = channels_dict.get(satelite)

    if hparams.wandb_model:
        artifact = wandb_logger.use_artifact(hparams.wandb_model, "model")
        model_path = artifact.file()
        model = model.load_from_checkpoint(model_path, spectral_num=num_channels)
    elif hparams.ckpt:
        try:
            model = model.load_from_checkpoint(hparams.ckpt, spectral_num=num_channels)
        except:
            model = model(num_channels)
            model.load_state_dict(torch.load(hparams.ckpt))
    else:
        try:
            model = model.load_from_checkpoint(weights_path, spectral_num=num_channels, satellite = satelite)
        except:
            model = model(num_channels, satellite = satelite)
            model.load_state_dict(torch.load(weights_path))
    
    datamodule = PANDataModule(data_dir, img_scale = img_scale, highpass = highpass, num_workers = 2, shuffle_train = False, batch_size = 1)

    if hparams.data == "rr":
        dataloader = datamodule.test_dataloader()
    elif hparams.data == "fr":
        dataloader = datamodule.predict_dataloader()

    results = trainer.predict(model, dataloader)

    os.makedirs(f"{hparams.outdir}/{model_name}", exist_ok=True)
    
    for index ,result in enumerate(results):
        result = result[slice].numpy().transpose(0,2,3,1)*img_scale
        generate_image_out(result, index, model_name)

def generate_image_out (image_out, batch_n, model_name):
    count = batch_n * image_out.shape[0]
    for index, image in enumerate(image_out):    
        path_out = f"out/{model_name}/image_out_{count + index}.png"
        image = linstretch(image)*255
        cv.imwrite(path_out, image)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--satellite", default="wv3")
    parser.add_argument("--img_scale", default=2047.0)
    parser.add_argument("--data_dir", default="./data/mat/wv3")
    parser.add_argument("--outdir", default="./out")
    parser.add_argument("--method", default="fusionnet", choices=["apnn", "bdpn", "dicnn", "drpnn", "fusionnet", "msdcnn", "pannet", "pnn", "mambfuse"])
    parser.add_argument("--wandb_model", default=None)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--data", default="fr", choices=["rr", "fr"])
    args = parser.parse_args()

    main(args)