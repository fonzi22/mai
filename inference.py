import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# tf.config.experimental.set_virtual_device_configuration(
#     gpus[0],
#     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=21000)]  
# )
# tf.config.set_visible_devices([], 'GPU')
from models import MobilenetV3, Resnet101, EfficientNetV2
from utils import *
import argparse
import configs
import os
from tensorflow.keras import layers
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras import Model
from tensorflow.keras.applications import MobileNetV3Large
import imageio
import numpy as np
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str,
                        help="Model name: [resnet101, mobilenetv3]")
    
    args = parser.parse_args()
    if args.model == "resnet101":
        import configs.resnet101_config as cfg
        model = Resnet101(cfg.num_classes, cfg.url)
    if args.model == "efficientnetv2":
        import configs.efficientnetv2_config as cfg
        model = EfficientNetV2(cfg.num_classes, (cfg.crop_size, cfg.crop_size, 3))
    elif args.model == "mobilenetv3":
        import configs.mobilenetv3_config as cfg
        model = MobilenetV3(cfg.num_classes, (cfg.crop_size, cfg.crop_size, 3))
        
    if cfg.backbone_trainable:
        model.backbone.trainable = True
    else:
        model.backbone.trainable = False
        
    if isinstance(cfg.lr, list):
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(cfg.boundaries, cfg.lr)
        optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=cfg.weight_decay)
    else:
        optimizer = tf.keras.optimizers.AdamW(learning_rate=cfg.lr, weight_decay=cfg.weight_decay)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    
    # # Tạo checkpoint để lưu cả model và optimizer
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpoint_dir = os.path.join(cfg.save_path, args.model + "")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    
    # Load checkpoint
    # if cfg.resume:
    model.build((None, cfg.crop_size, cfg.crop_size, 3))
    # model.load_weights("./save_model/mobilenetv3_auto_aug/best_model.weights.h5")
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        checkpoint.restore(latest_checkpoint).expect_partial()
        print(f"Loaded checkpoint from {latest_checkpoint}")
    else:
        print("No checkpoint found.")
    
    test_dataset = get_dataset(cfg, "./data/camera_scene_detection_validation/images", mode="test")
    output_file = "submission/results-resnet-pseudo.txt"
    

    with open(output_file, "w") as f:
        for i in tqdm(range(len(test_dataset))):
            inputs = test_dataset[i][0]
            predictions = model(inputs, training=False)
            predicted_classes = tf.argmax(predictions, axis=-1)
            for class_id in predicted_classes.numpy():
                f.write(str(class_id+1) + "\n")
                

    
    
