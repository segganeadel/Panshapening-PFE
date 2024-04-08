from models.apnn import APNN
from models.bdpn import BDPN
from models.dicnn import DICNN
from models.drpnn import DRPNN
from models.fusionnet import FusionNet
from models.msdcnn import MSDCNN
from models.pannet import PanNet
from models.pnn import PNN

import torch
from dataloader import Dataset_h5py_train
import pytorch_lightning as pl
import os
from torchvision.utils import save_image

data_path2 = os.path.join(".","data","h5py","qb","reduced_examples","test_qb_multiExm1.h5")
data = Dataset_h5py_train(data_path2, img_scale=2047.0, highpass=False)
data_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

model = APNN(4)
trainer = pl.Trainer()

predicted = trainer.predict(model, data_loader)

for predict in enumerate(predicted):
    save_image()
    break