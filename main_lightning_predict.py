from argparse import ArgumentParser

from models_lightning.apnn import APNN
from models_lightning.bdpn import BDPN
from models_lightning.dicnn import DICNN
from models_lightning.drpnn import DRPNN
from models_lightning.fusionnet import FusionNet
from models_lightning.msdcnn import MSDCNN
from models_lightning.pannet import PanNet
from models_lightning.pnn import PNN

import torch
import os
from datamodule_mat import PANDataModule
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger, CSVLogger

import cv2 as cv


def main(hparams):
    
    models = {
        "apnn":(APNN,"apnn.pth",False),
        "bdpn":(BDPN,"bdpn.pth",False),
        "dicnn":(DICNN,"dicnn1.pth",False),
        "drpnn":(DRPNN,"drpnn.pth",False),
        "fusionnet":(FusionNet,"fusionnet.pth",False),
        "msdcnn":(MSDCNN,"msdcnn.pth",False),
        "pannet":(PanNet,"panet.pth",True),
        "pnn":(PNN,"pnn.pth",False)
    }

    model_name = "pnn" # "apnn", "bdpn", "dicnn", "drpnn", "fusionnet", "msdcnn", "pannet", "pnn"
    model, weights_path, highpass = models.get(model_name)
    weights_path = os.path.join(".", "weights", "QB", weights_path)

    satelite = "qb"
    data_dir = os.path.join(".","data","mat",satelite)   
    datamodule = PANDataModule(data_dir, img_scale = 2047.0, highpass = highpass, num_workers = 7, shuffle_train = False, batch_size = 1)

    wandb_logger = WandbLogger(name=model_name, project="PanSharpening", prefix=satelite)
    csv_logger = CSVLogger(".")
    trainer = Trainer(logger=[wandb_logger, csv_logger], 
                      max_epochs=3)
    
    num_channels = 4 if satelite == "qb" else 8

    model = model.load_from_checkpoint("./PanSharpening/zyt6muug/checkpoints/epoch=9-step=2680.ckpt", spectral_num=num_channels)
    # model = model(num_channels)
    # model.load_state_dict(torch.load(weights_path))

    results = trainer.predict(model, datamodule)

    for index ,result in enumerate(results):
        print(result.shape)
        result = result[:,:3].numpy().transpose(0,2,3,1)*255
        generate_image_out(result, index, model_name)

def generate_image_out (image_out, batch_n, model_name):
    count = batch_n * image_out.shape[0]
    for index, image in enumerate(image_out):    
        print(image.shape, "image_out")
        print(image.max())
        path_out = f"out/image_out_{count + index}.png"
        cv.imwrite(path_out, image)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default=None)
    parser.add_argument("--devices", default=None)
    args = parser.parse_args()

    main(args)