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
        "mambfuse": (MambFuse,  "",             False)
    }

    # Choose the model
    model_name = hparams.method
    model, weights_path, highpass = models.get(model_name)

    satelite = hparams.satellite
    data_dir = hparams.data_dir

    wandb_logger = WandbLogger(name=model_name, project="PanSharpening", prefix = satelite, job_type="train", group = "mymodel", log_model="all")
    csv_logger = CSVLogger(".")
    trainer = Trainer(logger=[wandb_logger, csv_logger], 
                      max_epochs=int(hparams.epochs))
    
    num_channels = 4 if satelite == "qb" else 8
    model = model(spectral_num=num_channels) # 4 Channels if qb 8 for else
    datamodule = PANDataModule(data_dir, img_scale = 2047.0, highpass = highpass, num_workers = 2, shuffle_train = False, batch_size = 32)
    trainer.fit(model, datamodule)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--satellite", default="qb")
    parser.add_argument("--data_dir", default="./data/mat/qb")
    parser.add_argument("--method", default="mambfuse", choices=["apnn", "bdpn", "dicnn", "drpnn", "fusionnet", "msdcnn", "pannet", "pnn", "mambfuse"])
    parser.add_argument("--epochs", default= 100)
    args = parser.parse_args()

    main(args)