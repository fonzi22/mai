import os
import numpy as np
from PIL import Image, ImageEnhance
import tensorflow as tf
import random
import math

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
        self.on_epoch_end()  

    def __len__(self):
        return math.ceil(len(self.image_paths) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.image_paths[idx * self.batch_size : (idx + 1) * self.batch_size]
        if self.labels:
            batch_y = self.labels[idx * self.batch_size : (idx + 1) * self.batch_size]
            batch_y = np.array(batch_y, dtype=np.int32)
        else: 
            batch_y = None
        images = []
        for img_path in batch_x:
            img = Image.open(img_path).convert('RGB')
            img = img.resize(self.resize_size)
            img = self.random_crop(img, self.crop_size)
            # img = tf.image.random_flip_left_right(img)
            # if self.auto_augment:
            #     img = tfa.image.autoaugment(img, policy=tfa.image.autoaugment_policy.ImageNet)
            img = np.array(img, dtype=np.float32)
            # Chuẩn hoá ảnh về [-1, +1] 
            img = (img / 127.5) - 1.0
            images.append(img)
        
        images = np.stack(images, axis=0)  # [batch_size, H, W, 3]

        return images, batch_y

    def on_epoch_end(self):
        # Shuffle lại data sau mỗi epoch
        if self.shuffle:
            indices = np.arange(len(self.image_paths))
            np.random.shuffle(indices)
            self.image_paths = [self.image_paths[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]

    def random_crop(self, img, crop_size):
        """Hàm crop ngẫu nhiên crop_size=(cw, ch) từ ảnh PIL."""
        w, h = img.size
        cw, ch = crop_size
        if (w == cw) and (h == ch):
            return img  # Không cần crop
        if w < cw or h < ch:
            return img.resize(crop_size)
        # Tính toạ độ crop ngẫu nhiên
        left = np.random.randint(0, w - cw + 1)
        top = np.random.randint(0, h - ch + 1)
        right = left + cw
        bottom = top + ch
        return img.crop((left, top, right, bottom))
    
    # def get_transfrom(self):
        

# train_dataset = Dataset(
#     image_paths=train_image_paths,
#     labels=train_labels,
#     batch_size=32,
#     resize_size=(256, 256),
#     crop_size=(224, 224),
#     num_classes=30,
#     shuffle=True
# )

# val_dataset = Dataset(
#     image_paths=val_image_paths,
#     labels=val_labels,
#     batch_size=32,
#     resize_size=(256, 256),
#     crop_size=(224, 224),
#     num_classes=30,
#     shuffle=False
# )
