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
        "pnn":      (PNN,       "pnn.pth",      False),
        "mambfuse": (MambFuse,  "mambfuse.ckpt",False)
    }
    
    model_name = hparams.method
    satelite = hparams.satellite # mtf support ["qb", "ikonos", "geoeye1", "wv2", "wv3", "wv4"]
    data_dir = hparams.data_dir

    model, weights_path, highpass = models.get(model_name)
    weights_path = os.path.join(".", "weights", "QB", weights_path)

    wandb_logger = WandbLogger(name=model_name, project="PanSharpening", prefix = satelite, job_type="test", group = "mine")
    csv_logger = CSVLogger(".")
    trainer = Trainer(logger=[wandb_logger, csv_logger], 
                      devices=1, 
                      num_nodes=1)

    num_channels = 4 if satelite == "qb" else 8

    if hparams.wandb_model:
        artifact = wandb_logger.use_artifact(hparams.wandb_model, "model")
        model_path = artifact.file()
        model = model.load_from_checkpoint(model_path, spectral_num=num_channels, satellite = satelite)
    elif hparams.ckpt:
        try:
            model = model.load_from_checkpoint(hparams.ckpt, spectral_num=num_channels, satellite = satelite)
        except:
            model = model(num_channels, satellite = satelite)
            model.load_state_dict(torch.load(hparams.ckpt))
    else:
        try:
            model = model.load_from_checkpoint(weights_path, spectral_num=num_channels, satellite = satelite)
        except:
            model = model(num_channels, satellite = satelite)
            model.load_state_dict(torch.load(weights_path))
    
    datamodule = PANDataModule(data_dir, img_scale = 2047.0, highpass = highpass, num_workers = 3, shuffle_train = False, batch_size = 1)
    
    if hparams.data == "rr":
        dataloader = datamodule.test_dataloader()
    elif hparams.data == "fr":
        dataloader = datamodule.predict_dataloader()

    trainer.test(model, dataloader)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--satellite", default="qb")
    parser.add_argument("--data_dir", default="./data/mat/qb")
    parser.add_argument("--method", default="pnn", choices=["apnn", "bdpn", "dicnn", "drpnn", "fusionnet", "msdcnn", "pannet", "pnn", "mambfuse"])
    parser.add_argument("--wandb_model", default=None)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--data", default="rr", choices=["rr", "fr"])
    args = parser.parse_args()

    main(args)