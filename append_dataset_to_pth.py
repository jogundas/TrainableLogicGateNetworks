# /// script
# dependencies = [
#   "python-dotenv",
#   "numpy",
#   "wandb",
#   "torch",
#   "torchvision",
# ]
# [tool.uv]
# exclude-newer = "2024-02-20T00:00:00Z"
# ///

# import random
import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
# from datetime import datetime
# import time
import torch.profiler
# import hashlib
# import os
# import math
# import socket
# from zoneinfo import ZoneInfo
# import wandb
from lgn_model import Model

############################ CONFIG ########################

from dotenv import dotenv_values
config = dotenv_values(".env")

MODEL_FILENAME = "20250226-152349_binTestAcc7858_seed97798_epochs5_1x300_b256_lr10.pth"

BINARIZE_IMAGE_TRESHOLD = float(config.get("BINARIZE_IMAGE_TRESHOLD", 0.75))
IMG_WIDTH = int(config.get("IMG_WIDTH", 16))
INPUT_SIZE = IMG_WIDTH * IMG_WIDTH
DATA_SPLIT_SEED = int(config.get("DATA_SPLIT_SEED", 42))
TRAIN_FRACTION = float(config.get("TRAIN_FRACTION", 0.9))
NUMBER_OF_CATEGORIES = int(config.get("NUMBER_OF_CATEGORIES", 10))

try:
    device = torch.device("cuda" if torch.cuda.is_available() else 
                      "mps" if torch.backends.mps.is_available() else 
                      "cpu")
except:
    device = "cpu"


# ############################ DATA ########################

# ### GENERATORS
def binarize_image_with_histogram(image, verbose=False):
    threshold = torch.quantile(image, BINARIZE_IMAGE_TRESHOLD)
    return (image > threshold).float()

transform = transforms.Compose([
    transforms.Resize((IMG_WIDTH, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1)),
    transforms.Lambda(lambda x: binarize_image_with_histogram(x))
])

train_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    transform=transform,
    download=True
)


train_size = int(TRAIN_FRACTION * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(DATA_SPLIT_SEED))

### MOVE VAL DATASET TO GPU ###

val_dataset_samples = len(val_dataset)

val_images = torch.empty((val_dataset_samples, INPUT_SIZE), dtype=torch.float32, device=device)
val_labels = torch.empty((val_dataset_samples, NUMBER_OF_CATEGORIES), dtype=torch.float32, device=device)

val_labels_ = torch.empty((val_dataset_samples), dtype=torch.long, device=device)
for i, (image, label) in enumerate(val_dataset):
    val_images[i] = image
    val_labels_[i] = label
val_labels = torch.nn.functional.one_hot(val_labels_, num_classes=NUMBER_OF_CATEGORIES)
val_labels = val_labels.type(torch.float32)


def get_binarized_model(model, bin_value=1):
    model_binarized = Model(seed=model.seed, net_architecture=model.net_architecture, number_of_categories=model.number_of_categories, input_size=model.input_size).to(device)
    model_binarized.load_state_dict(model.state_dict())

    for layer_idx in range(0, len(model_binarized.layers)):
        ones_at = torch.argmax(model_binarized.layers[layer_idx].w.data, dim=0)
        model_binarized.layers[layer_idx].w.data.zero_()
        model_binarized.layers[layer_idx].w.data.scatter_(dim=0, index=ones_at.unsqueeze(0), value=bin_value)

        ones_at = torch.argmax(model_binarized.layers[layer_idx].c.data, dim=0)
        model_binarized.layers[layer_idx].c.data.zero_()
        model_binarized.layers[layer_idx].c.data.scatter_(dim=0, index=ones_at.unsqueeze(0), value=bin_value)

        model_binarized.layers[layer_idx].binarized = True

    return model_binarized





model = Model(seed=1, net_architecture=[300], number_of_categories=10, input_size=256).to(device)
model.load_state_dict(torch.load(MODEL_FILENAME, map_location=torch.device(device), weights_only=False))

binarized_model = get_binarized_model(model)

with torch.no_grad():
    X = val_images
    for layer_idx in range(0, len(binarized_model.layers)):
        X = binarized_model.layers[layer_idx](X)

    model.dataset_input = val_images
    model.dataset_output = X

filename_with_data = MODEL_FILENAME[:-4] + "_withData.pth"
print(f"New filename: {filename_with_data}")
torch.save(model.state_dict(), filename_with_data)