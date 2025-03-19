import os 
import json
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

import configs.mobilenetv3_config as cfg
from models import MobilenetV3
from data.dataloader import SemiSupervisedDataset
from data.augmentations import *  # Assuming you have some standard augmentations here
from utils import *               # For clean_old_checkpoints, etc.
from evaluate import evaluate     # For test set evaluation
from algorithms.fullmatch import fullmatch_train_step

# Prepare Datasets
dataset = SemiSupervisedDataset(
    data_dir=cfg.train_folder, 
    label_file=cfg.label_file, 
    split_ratio=cfg.labeled_ratio, 
    image_size=(cfg.crop_size, cfg.crop_size),
    batch_size=cfg.batch_size,
    shuffle=True
)
labeled_ds = dataset.get_labeled_dataset()     # => yields (images, labels)
unlabeled_ds = dataset.get_unlabeled_dataset() # => yields (images, _)
test_dataset = get_dataset(cfg, None, 'test')  # your custom function

for step, (inputs, labels) in enumerate(labeled_ds):
    print(inputs, labels)
    break