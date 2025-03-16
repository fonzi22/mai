import os
import glob
import json
from dataset import Dataset
import gc
import tensorflow as tf
from keras import backend as K

def read_json(file_path):
    with open(file_path, 'r') as file:
        content = json.load(file)
    return content

def load_data(cfg, folder_path):
    image_paths = []
    labels = []
    label_mapping = read_json(cfg.label_file)
    

    for c in os.listdir(folder_path):
        for file in os.listdir(os.path.join(folder_path, c)):
            image_paths.append(os.path.join(folder_path, c, file))
            labels.append(label_mapping[c])
    return image_paths, labels

def get_dataset(cfg, folder_path, mode='train'):
    image_paths = []
    if mode == 'test':
        for file in sorted(os.listdir(folder_path), key=lambda x: int(x.split('.')[0])):
            image_paths.append(os.path.join(folder_path, file))
        dataset = Dataset(
            image_paths=image_paths,
            labels = None,
            batch_size=cfg.batch_size,
            resize_size=cfg.resize_size,
            crop_size=cfg.crop_size,
            num_classes=cfg.num_classes,
            shuffle=False,
        )
    elif mode == 'train':
        labels = []
        label_mapping = read_json(cfg.label_file)
        for c in os.listdir(folder_path):
            for file in os.listdir(os.path.join(folder_path, c)):
                image_paths.append(os.path.join(folder_path, c, file))
                labels.append(label_mapping[c])
        if cfg.pseudo:
            data = read_json(cfg.pseudo_path)
            for x in data:
                image_paths.append(x['image_path'])
                labels.append(x['label'])
        dataset= Dataset(
            image_paths=image_paths,
            labels = labels,
            batch_size=cfg.batch_size,
            resize_size=cfg.resize_size,
            crop_size=cfg.crop_size,
            num_classes=cfg.num_classes,
            shuffle=True,
            auto_augment=cfg.auto_augment
        )
    return dataset
    

class ClearMemory(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        tf.keras.backend.clear_session()
        

def clean_old_checkpoints(checkpoint_dir, max_to_keep=3):
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "ckpt-*")), key=os.path.getmtime)
    if len(checkpoints) > max_to_keep:
        for ckpt in checkpoints[:-max_to_keep]:
            os.remove(ckpt)
            print(f"Deleted old checkpoint: {ckpt}")