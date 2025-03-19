import tensorflow as tf
import pathlib
import random
import math
import math
import os
from collections import defaultdict
from utils import read_json

class SemiSupervisedDataset:
    def __init__(self, data_dir, label_file, split_ratio=0.2, image_size=(160, 160), batch_size=256, shuffle=True):
        self.data_dir = pathlib.Path(data_dir)
        self.label_file = label_file
        self.split_ratio = split_ratio 
        self.image_size = image_size
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.image_paths = []
        self.labels = []

        # Example read_json (placeholder) - ensure this matches your actual implementation
        # label_file could be a JSON with { "class_folder_name": integer_label, ... }
        label_mapping = read_json(label_file)
        
        # Collect images & labels
        for c in os.listdir(data_dir):
            class_dir = os.path.join(data_dir, c)
            if not os.path.isdir(class_dir):
                continue
            for file in os.listdir(class_dir):
                image_path = os.path.join(class_dir, file)
                self.image_paths.append(image_path)
                self.labels.append(label_mapping[c])

        # Split data into labeled and unlabeled
        self.labeled_images, self.labeled_labels, self.unlabeled_images, self.unlabeled_labels = self._split_data()

    def _split_data(self):
        """Split data into labeled and unlabeled subsets by ratio."""
        label_groups = defaultdict(list)
        for path, lbl in zip(self.image_paths, self.labels):
            label_groups[lbl].append(path)
        
        labeled_subset = []
        labeled_labels = []
        unlabeled_subset = []
        unlabeled_labels = []
        
        for lbl, paths in label_groups.items():
            random.shuffle(paths)
            num_labeled = math.ceil(len(paths) * self.split_ratio)
            labeled_paths = paths[:num_labeled]
            unlabeled_paths = paths[num_labeled:]
            labeled_subset.extend(labeled_paths)
            unlabeled_subset.extend(unlabeled_paths)
            labeled_labels.extend([lbl] * len(labeled_paths))
            unlabeled_labels.extend([lbl] * len(unlabeled_paths))
        
        return labeled_subset, labeled_labels, unlabeled_subset, unlabeled_labels

    def _load_image(self, image_path, label):
        """Read, resize, and normalize an image."""
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, self.image_size)
        image = (image / 127.5) - 1.0
        return image, label

    def get_labeled_dataset(self):
        """Create a tf.data.Dataset from labeled data."""
        dataset = tf.data.Dataset.from_tensor_slices((self.labeled_images, self.labeled_labels))
        dataset = dataset.map(lambda x, y: self._load_image(x, y), num_parallel_calls=tf.data.AUTOTUNE)
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=len(self.labeled_images))
        dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

    def get_unlabeled_dataset(self):
        """Create a tf.data.Dataset from unlabeled data (images only)."""
        # We'll still keep the label in the pipeline to track which class folder they came from
        # (though unlabeled means we won't use it directly for supervised loss).
        dataset = tf.data.Dataset.from_tensor_slices((self.unlabeled_images, self.unlabeled_labels))
        dataset = dataset.map(lambda x, y: self._load_image(x, y), num_parallel_calls=tf.data.AUTOTUNE)
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=len(self.unlabeled_images))
        dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

