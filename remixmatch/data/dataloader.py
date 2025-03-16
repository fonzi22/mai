import tensorflow as tf
import numpy as np
import os
from .augmentations import get_weak_augmentation, get_strong_augmentation, get_rotation_augmentation

class SemiSupervisedDataLoader:
    def __init__(self, args):
        self.args = args
        self.num_classes = args.num_classes
        self.num_labeled = args.num_labeled
        self.batch_size = args.batch_size
        self.eval_batch_size = args.eval_batch_size
        self.uratio = args.uratio
        self.img_size = args.img_size
        self.rot_loss_ratio = args.rot_loss_ratio

        self._load_custom_dataset()

        self.class_distribution = self._compute_class_distribution()
    
    def 