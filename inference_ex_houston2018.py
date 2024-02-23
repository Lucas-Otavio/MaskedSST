import os
import sys
import random
import warnings

import wandb
import torch
import numpy as np
from tqdm import tqdm
from torchmetrics import Accuracy
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import rasterio
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

from src.vit_spatial_spectral import ViTSpatialSpectral
from src.utils import (
    get_supervised_data,
    load_checkpoint,
    get_finetune_config,
    get_val_epochs,
    train_step,
)
from src.data_enmap import dfc_labels_train, StandardizeEnMAP
from src.data_houston2018 import StandardizeHouston2018, labels as houston2018_labels, Houston2018Dataset, Houston2018LabelTransform

import torchvision
from src.data_enmap import ToTensor



SEED = 5
dataset_name = "houston2018"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"device: {device}")

valid_datasets = ["enmap", "houston2018"]

assert (
    dataset_name in valid_datasets
), f"Please provide a valid dataset name from {valid_datasets}, provided: {dataset_name=}"

config = get_finetune_config(
    f"configs/finetune_config_{dataset_name}.yaml",
    "configs/config.yaml",
    SEED,
    device,
)

model = ViTSpatialSpectral(
            image_size=config.image_size - config.patch_sub,
            spatial_patch_size=config.patch_size,
            spectral_patch_size=config.band_patch_size,
            num_classes=config.n_classes,
            dim=config.transformer_dim,
            depth=config.transformer_depth,
            heads=config.transformer_n_heads,
            mlp_dim=config.transformer_mlp_dim,
            dropout=config.transformer_dropout,
            emb_dropout=config.transformer_emb_dropout,
            channels=config.n_bands,
            spectral_pos=config.spectral_pos,
            spectral_pos_embed=config.spectral_pos_embed,
            blockwise_patch_embed=config.blockwise_patch_embed,
            spectral_only=config.spectral_only,
            pixelwise=config.pixelwise,
            pos_embed_len=config.pos_embed_len,
        )
print(f"Model parameters: {sum([p.numel() for p in model.parameters()]):,}")

print(f"checkpoint path: {config.checkpoint_path}")

#---------------- # Added part # --------------- #

houston = Houston2018Dataset(
            config.train_path,
            config.train_label_path,
            torchvision.transforms.Compose(
        [
            # todo: rotate
            standardizer,
            ToTensor(),
        ]
    ),
            Houston2018LabelTransform(),
            patch_size=config.image_size - config.patch_sub,
            test=False,
            drop_unlabeled=True,
            fix_train_patches=False,
            pixelwise=config.pixelwise,
            rgb_only=config.rgb_only,
        )

scene = houston.img[:,:524,:524]
label_map = houston.label[:524,:524]
eff_size = config.image_size - config.patch_sub
predicted_map = torch.zeros(label_map.shape)

for x in tqdm(range(0, 524)):
    for y in range(0, 524):
        img = scene[:,x:x + eff_size, y:y + eff_size].unsqueeze(0) #.narrow(2, x, eff_size).narrow(3, y, eff_size)
        if x + eff_size >= 524 or y + eff_size >= 524:
            continue
                
        output = model(img)

        pred = output.argmax(dim=1)

        #pred[eff_size//2, eff_size//2]
        predicted_map[x + eff_size//2, y + eff_size//2] = pred[0, eff_size//2, eff_size//2]

torch.save(predicted_map, "predicted_map.pt")