import os
import numpy as np
from PIL import Image
import tensorflow as tf
import random
import math
from randaugment import RandAugment

class Dataset(tf.keras.utils.Sequence):
    def __init__(self,
                 image_paths,
                 labels,
                 batch_size=32,
                 resize_size=256,
                 crop_size=224,
                 num_classes=30,
                 shuffle=True,
                 auto_augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.resize_size = (resize_size, resize_size)
        self.crop_size = (crop_size, crop_size)
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.auto_augment = auto_augment
        self.strong_augment = RandAugment(n=2, m=5)  
        self.on_epoch_end()  

    def __len__(self):
        return math.ceil(len(self.image_paths) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.image_paths[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_y = np.array(self.labels[idx * self.batch_size : (idx + 1) * self.batch_size], dtype=np.int32) if self.labels else None
        
        images = []
        for img_path in batch_x:
            img = Image.open(img_path).convert('RGB')
            img = img.resize(self.resize_size)
            img = self.random_crop(img, self.crop_size)
            
            if self.auto_augment:
                # Chọn ngẫu nhiên giữa Weak hoặc Strong Augment
                if random.random() > 0.5:  
                    img = self.weak_augment(img)  # Flip hoặc crop nhẹ
                else:
                    img = self.strong_augment(np.array(img))  # RandAugment
            
            img = np.array(img, dtype=np.float32)  
            img = (img / 127.5) - 1.0  # Chuẩn hóa về [-1,1]
            images.append(img)
        
        return np.stack(images, axis=0), batch_y

    def on_epoch_end(self):
        if self.shuffle:
            indices = np.arange(len(self.image_paths))
            np.random.shuffle(indices)
            self.image_paths = [self.image_paths[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]

    def random_crop(self, img, crop_size):
        """Crop ngẫu nhiên crop_size=(cw, ch) từ ảnh PIL."""
        w, h = img.size
        cw, ch = crop_size
        if w < cw or h < ch:
            return img.resize(crop_size)
        left = np.random.randint(0, w - cw + 1)
        top = np.random.randint(0, h - ch + 1)
        return img.crop((left, top, left + cw, top + ch))

    def weak_augment(self, img):
        """Áp dụng các biến đổi nhẹ (Weak Augment)"""
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img
