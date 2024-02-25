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

standardizer = StandardizeHouston2018()

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

img_test1 = houston.img[:, :, :596]
img_test2 = houston.img[:, :601, 596:2980]
img_test3 = houston.img[:, :, 2980:]

label_test1 = houston.label[:, :596]
label_test2 = houston.label[:601, 596:2980]
label_test3 = houston.label[:, 2980:]


def generate_img(img, label, save_code, batch_size=64):
    scene = img
    label_map = label
    eff_size = config.image_size - config.patch_sub
    predicted_map = torch.zeros(label_map.shape)
    print(f"Generating image: {save_code}")

    size_x, size_y = img.shape[1]-eff_size+1, img.shape[2]-eff_size+1
    #images_cropped = torch.zeros((size_y, 50, 8, 8))
    for x in tqdm(range(0, size_x)):
        for batch_begin in range(0, size_y, batch_size):
            size_batch = min(batch_size, size_y-batch_begin)
            images_cropped = torch.zeros((size_batch, 50, 8, 8))
            for y in range(batch_begin, batch_begin+size_batch):
                img = scene[:,x:x + eff_size, y:y + eff_size]
                images_cropped[y-batch_begin] = img

            output = model(images_cropped)
            pred = output.argmax(dim=1)
            #print(predicted_map[x + eff_size//2, eff_size//2 : eff_size//2 + size_y].shape, pred[:, eff_size//2, eff_size//2].shape)
            predicted_map[x + eff_size//2, batch_begin + eff_size//2 : batch_begin + eff_size//2 + size_batch] = pred[:, eff_size//2, eff_size//2]
    
    torch.save(predicted_map, f"predicted_map_{save_code}.pt")
            

#generate_img(houston.img[:,:65,:65], houston.label[:65,:65], "65_65")
#generate_img(houston.img[:,:32,:64], houston.label[:32,:64], "32_64")
#generate_img(img_test1, label_test1, "test1")
#generate_img(img_test2, label_test2, "test2")
#generate_img(img_test3, label_test3, "test3")