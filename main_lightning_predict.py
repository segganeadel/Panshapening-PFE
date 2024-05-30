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

import torch
from datamodule_mat import PANDataModule
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from scipy.io import savemat

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

    model_name = "fusionnet" # "apnn", "bdpn", "dicnn", "drpnn", "fusionnet", "msdcnn", "pannet", "pnn"
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

    # model = model.load_from_checkpoint("./PanSharpening/hbqnqyh9/checkpoints/epoch=9-step=5360.ckpt", spectral_num=num_channels)
    model = model(num_channels)
    model.load_state_dict(torch.load(weights_path))
    
    test_dataloader = datamodule.test_dataloader()
    results = trainer.predict(model, test_dataloader)
    os.makedirs("out", exist_ok=True)

    for index ,result in enumerate(results):
        result = result * 2047.0
        result = result.squeeze(0)
        print(result.shape)
        savemat(f"out/{index}.mat", {"out": result})
        # result = result[:,:3].numpy().transpose(0,2,3,1)*255
        # generate_image_out(result, index, model_name)

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