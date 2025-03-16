import tensorflow as tf
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

def representative_dataset():
    dataset_size = 600
    for i in range(dataset_size):
        data = imageio.imread("data/camera_scene_detection_validation/images/" + str(i) + ".jpg")
        
        # Check if the image has 3 dimensions (height, width, channels)
        if data.ndim == 2:  # If the image is grayscale
            data = np.stack((data,)*3, axis=-1)  # Convert to RGB by stacking
        elif data.ndim == 3 and data.shape[2] == 4:  # If the image has an alpha channel
            data = data[..., :3]  # Remove the alpha channel
        
        # Resize the image to (128, 128)
        data = tf.image.resize(data, [128, 128])  # Resize to match input size
        data = np.reshape(data, [1, 128, 128, 3])  # Update reshape dimensions
        yield [data.astype(np.float32)]
    
def convert_to_tflite(latest_checkpoint, optimizer):
    # Define the input size, custom for MobileNetV3
    input_size = (128, 128, 3)

    # Input layer
    input_image = layers.Input(shape=input_size)
    input_image_normalized = preprocess_input(input_image)  # Normalize input

    # Initialize MobileNetV3Large (you can replace with MobileNetV3Small if needed)
    base_model = MobilenetV3(cfg.num_classes, (cfg.crop_size, cfg.crop_size, 3))

    # Freeze layers of the base model
    for layer in base_model.layers:
        layer.trainable = False
        
    
    model = Model(inputs=input_image, outputs=base_model(input_image_normalized))
    print(model.summary())

    checkpoint = tf.train.Checkpoint(model=base_model, optimizer=optimizer)
    checkpoint.restore(tf.train.latest_checkpoint(latest_checkpoint)).expect_partial()
    
    # Optionally load pre-trained weights for your custom model
    # print("Loading weights from checkpoint...", latest_checkpoint)
    # model.load_weights(latest_checkpoint)

    # Convert the model to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Enable optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Set representative dataset for INT8 quantization
    converter.representative_dataset = representative_dataset

    # Ensure only INT8 operations are used
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    # Set input and output data types to uint8
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    # Convert the model
    tflite_model = converter.convert()

    # Save the TFLite model to a file
    output_tfmodel = "submission/model.tflite"
    with open(output_tfmodel, "wb") as f:
        f.write(tflite_model)

    print("TFLite model conversion completed and saved as 'custom_mobilenetv3_model.tflite'")


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
    
    # Tạo checkpoint để lưu cả model và optimizer
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpoint_dir = os.path.join(cfg.save_path, args.model)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_prefix = os.path.join(checkpoint_dir, args.model + "_ckpt")

    # Change the checkpoint saving format
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
    
    # Load checkpoint
    if cfg.resume:
        model.build((None, cfg.crop_size, cfg.crop_size, 3))
        latest_checkpoint = checkpoint_manager.latest_checkpoint
        if latest_checkpoint:
            checkpoint.restore(latest_checkpoint).expect_partial()
            convert_to_tflite(latest_checkpoint, optimizer)
            print(f"Loaded checkpoint from {latest_checkpoint}")
        else:
            print("No checkpoint found.")
    
    test_dataset = get_dataset(cfg, "./data/camera_scene_detection_validation/images", mode="test")
    output_file = "submission/results.txt"
    

    with open(output_file, "w") as f:
        for i in range(len(test_dataset)):
            inputs = test_dataset[i][0]
            predictions = model(inputs, training=False)
            predicted_classes = tf.argmax(predictions, axis=-1)
            for class_id in predicted_classes.numpy():
                f.write(str(class_id+1) + "\n")
                

    
    
