from argparse import ArgumentParser

from models.mambfuse import MambFuse

import torch
import os
from datamodule_mat import PANDataModule
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from dotenv import load_dotenv

load_dotenv()
def main(hparams):
    

    satelite = "qb"
    highpass = False
    data_dir = os.path.join(".","data","mat",satelite)   
    datamodule = PANDataModule(data_dir, img_scale = 2047.0, highpass = highpass, num_workers = 3, shuffle_train = False, batch_size = 1)

    csv_logger = CSVLogger(".")
    trainer = Trainer(logger=[csv_logger], 
                      max_epochs=3)
    
    num_channels = 4 if satelite == "qb" else 8
    model = MambFuse(num_channels)

    # model = model.load_from_checkpoint("./PanSharpening/hbqnqyh9/checkpoints/epoch=9-step=5360.ckpt", spectral_num=num_channels)
    # model = model(num_channels)
    # model.load_state_dict(torch.load(weights_path))
    
    trainer.test(model, datamodule)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default=None)
    parser.add_argument("--devices", default=None)
    args = parser.parse_args()

    main(args)