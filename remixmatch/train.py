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

# Metamorphic relations & testing utilities
def metamorphic_relation_rotation(image, angle=0.1):
    """Rotate image slightly by `angle` radians (~5.73° if angle=0.1 rad)."""
    # Convert image to NumPy array if it's a TensorFlow tensor.
    if tf.is_tensor(image):
        image = image.numpy()
    angle_degrees = np.degrees(angle)
    transform = A.Rotate(limit=(angle_degrees, angle_degrees), p=1.0)
    augmented = transform(image=image)
    return augmented['image']



def metamorphic_relation_180_rotation(image, label):
    """
    Rotate image 180 degree.
    """
    if tf.is_tensor(image):
        image = image.numpy()
    
    transform = A.Rotate(limit=(180, 180), p=1.0)
    augmented = transform(image=image)
    rotated_image = augmented['image']
    
    return rotated_image, label

def run_metamorphic_tests(model, dataset, metamorphic_relations):
    """
    Run metamorphic tests on a (preferably small) dataset.
    For each sample, apply each relation and compare model predictions 
    to the 'expected_label'.
    Returns a list of (relation_name, pass_fail).
    
    NOTE: 'dataset' here should yield (image, label) for single samples
    or a small batch of size=1. 
    If your dataset is batched, you'll need to iterate differently.
    """
    test_results = []
    for image, label in dataset:
        # If dataset is batched, 'image' and 'label' might be a batch. 
        # For simplicity, assume batch_size=1 or do something like:
        #   for i in range(image.shape[0]):
        #       single_img = image[i]
        #       single_lbl = label[i]
        
        # Convert to rank-4 tensor if needed
        if len(image.shape) == 3:
            image = tf.expand_dims(image, axis=0)  # shape (1, H, W, C)

        for relation in metamorphic_relations:
            rname = relation.__name__
            if rname == "metamorphic_relation_180_rotation":
                # This relation returns (transformed_img, label)
                transformed_img, expected_label = relation(image[0], label)
                # Re-add batch dimension
                transformed_img = tf.expand_dims(transformed_img, axis=0)
            else:
                # e.g. metamorphic_relation_rotation
                transformed_img = relation(image[0])  # returns an image
                expected_label = label
                transformed_img = tf.expand_dims(transformed_img, axis=0)
            
            # Model prediction
            pred_logits = model(transformed_img, training=False)
            pred_label = tf.argmax(pred_logits, axis=-1).numpy()[0]
            
            pass_fail = (pred_label == expected_label)
            test_results.append((rname, pass_fail))
    return test_results

def extract_failed_relations(test_results):
    """
    Collect relation names that had any failure.
    test_results: list of (relation_name, pass_fail).
    """
    failed_relations = []
    for relation_name, passed in test_results:
        if not passed and relation_name not in failed_relations:
            failed_relations.append(relation_name)
    return failed_relations

def generate_augmented_dataset(original_dataset, selected_relations, batch_size=32):
    """
    For each sample in `original_dataset`, create additional samples 
    based on the chosen metamorphic relations. Return a new tf.data.Dataset.
    NOTE: 
      - If 'original_dataset' is batched, you'll need to unbatch or handle it carefully.
      - For 'metamorphic_relation_180_rotation', we expect it to return (img, label).
      - For 'metamorphic_relation_rotation', it returns just the image, label stays the same.
    """
    # Because we might have different shapes, let's collect them in memory first (not ideal for huge datasets).
    all_images = []
    all_labels = []

    # Map string relation -> actual function
    name_to_func = {
        "metamorphic_relation_180_rotation": metamorphic_relation_180_rotation,
        "metamorphic_relation_rotation": metamorphic_relation_rotation
    }

    for batch_images, batch_labels in original_dataset:
        # If this dataset is batched, we have multiple samples
        for i in range(batch_images.shape[0]):
            img = batch_images[i]
            lbl = batch_labels[i].numpy()
            
            # Always add the original
            all_images.append(img.numpy())
            all_labels.append(lbl)
            
            # For each failed relation, generate a new sample
            for rel_name in selected_relations:
                func = name_to_func[rel_name]
                if rel_name == "metamorphic_relation_180_rotation":
                    aug_img, aug_lbl = func(img, lbl)
                else:
                    # e.g. metamorphic_relation_rotation
                    aug_img = func(img)
                    aug_lbl = lbl
                
                all_images.append(aug_img.numpy())
                all_labels.append(aug_lbl)

    ds = tf.data.Dataset.from_tensor_slices((all_images, all_labels))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def main():
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
    test_dataset = get_dataset(cfg, '/home/s24gb-2/Desktop/GenAI4E/CVPR2025W/MAI_SceneDetection/data', 'test')  # your custom function

    # Build Model
    model = MobilenetV3(cfg.num_classes, (cfg.crop_size, cfg.crop_size, 3))
    model.backbone.trainable = False  # freeze backbone if desired

    # Define Optimizer, Loss, Metrics
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=cfg.lr, 
        weight_decay=cfg.weight_decay
    )
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    # Basic Train Step (supervised)
    @tf.function
    def train_step(inputs, labels):
        with tf.GradientTape() as tape:
            preds = model(inputs, training=True)
            loss = loss_fn(labels, preds)
            acc = accuracy_metric(labels, preds)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, acc

    # Checkpoint setup
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpoint_dir = os.path.join(cfg.save_path)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    # Logging setup
    logging.basicConfig(
        filename=os.path.join(checkpoint_dir, 'training.log'),
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )
    # Log config
    cfg_dict = {k: v for k, v in vars(cfg).items() if not k.startswith("__")}
    logging.info("Configuration settings:\n%s", json.dumps(cfg_dict, indent=4))

    # Optionally resume
    if cfg.resume:
        model.build((None, cfg.crop_size, cfg.crop_size, 3))
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            checkpoint.restore(latest_checkpoint).expect_partial()
            print(f"Loaded checkpoint from {latest_checkpoint}")
        else:
            print("No checkpoint found.")

    EPOCHS = cfg.epochs
    for epoch in range(1):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0

        for step, (inputs, labels) in enumerate(labeled_ds):
            loss, acc = train_step(inputs, labels)
            total_loss += loss.numpy()
            total_acc += acc.numpy()
            num_batches += 1

            print(f"  Step {step}: Loss = {loss.numpy():.4f}, Accuracy = {acc.numpy():.4f}")

        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_acc:.4f}")

        # Log to file
        logging.info(f"Epoch {epoch+1}/{EPOCHS} - Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_acc:.4f}")
        
        # Save checkpoint each epoch
        checkpoint.save(file_prefix=checkpoint_prefix)  
        clean_old_checkpoints(checkpoint_dir=checkpoint_dir, max_to_keep=3)

        # Evaluate every N epochs
        if (epoch + 1) % 5 == 0:
            test_loss, test_acc = evaluate(model, test_dataset)
            print(f"  Test - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
            logging.info(f"  Test - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")

    print("\nRunning Metamorphic Tests...")
    metamorphic_relations = [
        metamorphic_relation_rotation, 
        metamorphic_relation_180_rotation
    ]
    # Let's take e.g. 100 samples from labeled_ds
    # If your dataset is large, you might do: labeled_ds.unbatch().take(100)
    test_results = run_metamorphic_tests(model, labeled_ds, metamorphic_relations)
    failed_relations = extract_failed_relations(test_results)

    print(f"Failed metamorphic relations: {failed_relations}")
    logging.info(f"Failed metamorphic relations: {failed_relations}")

    # If any relations failed, create an augmented dataset & fine-tune
    if failed_relations:
        print("Generating augmented dataset with failed metamorphic relations...")
        augmented_ds = generate_augmented_dataset(labeled_ds, failed_relations, batch_size=cfg.batch_size)
        
        # Fine-tune for a few epochs on the augmented dataset
        fine_tune_epochs = 1
        for epoch in range(fine_tune_epochs):
            print(f"Fine-tune Epoch {epoch+1}/{fine_tune_epochs}")
            total_loss = 0.0
            total_acc = 0.0
            num_batches = 0

            for step, (inputs, labels) in enumerate(augmented_ds):
                loss, acc = train_step(inputs, labels)
                total_loss += loss.numpy()
                total_acc += acc.numpy()
                num_batches += 1

            avg_loss = total_loss / num_batches
            avg_acc = total_acc / num_batches
            print(f"  Fine-tune Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_acc:.4f}")
            logging.info(f"Fine-tune Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_acc:.4f}")
            
            checkpoint.save(file_prefix=os.path.join(checkpoint_dir, 'metamorphic', 'ckpt'))  
            clean_old_checkpoints(checkpoint_dir=os.path.join(checkpoint_dir, 'metamorphic'), max_to_keep=3)
            
            if (epoch + 1) % 5 == 0:
                test_loss, test_acc = evaluate(model, test_dataset)
                print(f"  Test - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
                logging.info(f"  Test - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
                
    # Now run FullMatch cycles on labeled & unlabeled data
    print("\nStarting FullMatch cycles...")
    p_cutoff = 0.95
    lambda_u = 1.0
    lambda_p = 0.5
    T = 0.5
    # Prepare batched datasets
    labeled_batches = labeled_ds.shuffle(1000).batch(cfg.batch_size)
    unlabeled_batches = unlabeled_ds.shuffle(1000).batch(cfg.batch_size)

    num_cycles = 1
    for cycle in range(num_cycles):
        print(f"\nFullMatch Cycle {cycle+1}/{num_cycles}")
        for (x_lb, y_lb), (x_ulb, _) in zip(labeled_batches, unlabeled_batches):
            sup_loss, unsup_loss, total_loss, mask_mean = fullmatch_train_step(
                model, optimizer, x_lb, y_lb, x_ulb, p_cutoff, lambda_u, lambda_p, T
            )
        print(f"Cycle {cycle+1} - Total Loss = {total_loss.numpy():.4f}, Mask Mean = {mask_mean.numpy():.4f}")
        
        checkpoint.save(file_prefix=os.path.join(checkpoint_dir, 'fullmatch', 'ckpt'))  
        clean_old_checkpoints(checkpoint_dir=os.path.join(checkpoint_dir, 'fullmatch'), max_to_keep=3)
        
        if (cycle + 1) % 5 == 0:
            test_loss, test_acc = evaluate(model, test_dataset)
            print(f"  Test - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
            logging.info(f"  Test - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
            
    # Final evaluation on test set
    print("\nEvaluating final model on test set...")
    test_loss, test_acc = evaluate(model, test_dataset)
    print(f"FullMatch Final Test Accuracy: {test_acc*100:.2f}%  (Loss: {test_loss:.4f})")

    # Save final model
    model.save("fullmatch_model.h5")
    print("Model saved as fullmatch_model.h5")


if __name__ == "__main__":
    main()
