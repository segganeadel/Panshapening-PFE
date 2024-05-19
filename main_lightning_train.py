from argparse import ArgumentParser

from models_lightning.apnn import APNN
from models_lightning.bdpn import BDPN
from models_lightning.dicnn import DICNN
from models_lightning.drpnn import DRPNN
from models_lightning.fusionnet import FusionNet
from models_lightning.msdcnn import MSDCNN
from models_lightning.pannet import PanNet
from models_lightning.pnn import PNN

import os
from datamodule_mat import PANDataModule
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger, CSVLogger

def main(hparams):
    
    models = {
    # "model_name": (model,     weights_path, highpass)
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
    model_name = "fusionnet" # "apnn", "bdpn", "dicnn", "drpnn", "fusionnet", "msdcnn", "pannet", "pnn"
    model, weights_path, highpass = models.get(model_name)

    satelite = "qb"
    data_dir = os.path.join(".","data","mat",satelite)   
    datamodule = PANDataModule(data_dir, img_scale = 2047.0, highpass = highpass, num_workers = 7, shuffle_train = False, batch_size = 32)

    wandb_logger = WandbLogger(name=model_name, project="PanSharpening", prefix = satelite, job_type="train", group = "dlpan")
    csv_logger = CSVLogger(".")
    trainer = Trainer(logger=[wandb_logger, csv_logger], 
                      max_epochs=10)
    
    num_channels = 4 if satelite == "qb" else 8
    model = model(spectral_num=num_channels) # 4 Channels if qb 8 for else
    trainer.fit(model, datamodule)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default=None)
    parser.add_argument("--devices", default=None)
    args = parser.parse_args()

    main(args)