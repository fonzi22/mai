import tensorflow as tf
import os
import logging
import argparse
import configs
from models import MobilenetV3, Resnet101, EfficientNetV2
from utils import *
from evaluate import evaluate

# Cấu hình GPU
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str,
                        help="Model name: [resnet101, mobilenetv3, efficientnetv2]")
    
    args = parser.parse_args()
    if args.model == "resnet101":
        import configs.resnet101_config as cfg
        model = Resnet101(cfg.num_classes, cfg.url)
    elif args.model == "efficientnetv2":
        import configs.efficientnetv2_config as cfg
        model = EfficientNetV2(cfg.num_classes, (cfg.crop_size, cfg.crop_size, 3))
    elif args.model == "mobilenetv3":
        import configs.mobilenetv3_config as cfg
        model = MobilenetV3(cfg.num_classes, (cfg.crop_size, cfg.crop_size, 3))
    else:
        raise ValueError("Model không hợp lệ!")

    # Thiết lập chế độ train của backbone
    if cfg.backbone_trainable:
        model.backbone.trainable = True
    else:
        model.backbone.trainable = False

    train_dataset = get_dataset(cfg, cfg.train_folder, mode='train')
    test_dataset = get_dataset(cfg, '/home/s48gb/Desktop/GenAI4E/mai/data', 'test')  
    
    if isinstance(cfg.lr, list):
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(cfg.boundaries, cfg.lr)
        optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=cfg.weight_decay)
    else:
        optimizer = tf.keras.optimizers.AdamW(learning_rate=cfg.lr, weight_decay=cfg.weight_decay)
    
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    # Tạo checkpoint để lưu model và optimizer
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpoint_dir = os.path.join(cfg.save_path, args.model)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    
    # Cấu hình logging
    logging.basicConfig(
        filename=os.path.join(checkpoint_dir, 'training.log'),         # File log sẽ được lưu tại đây
        level=logging.INFO,              # Mức log INFO và cao hơn sẽ được ghi
        format='%(asctime)s:%(levelname)s:%(message)s'
    )
    
    # Ghi log cấu hình cfg 
    cfg_dict = {k: v for k, v in vars(cfg).items() if not k.startswith("__")}
    logging.info("Configuration settings:\n%s", json.dumps(cfg_dict, indent=4))

    # Load checkpoint nếu cần
    if cfg.resume:
        model.build((None, cfg.crop_size, cfg.crop_size, 3))
        latest_checkpoint = tf.train.latest_checkpoint('/home/s48gb/Desktop/GenAI4E/mai/save_model/mobilenetv3_pseudo')
        if latest_checkpoint:
            checkpoint.restore(latest_checkpoint).expect_partial()
            print(f"Loaded checkpoint from {latest_checkpoint}")
        else:
            print("No checkpoint found.")

    @tf.function
    def train_step(inputs, labels):
        with tf.GradientTape() as tape:
            preds = model(inputs, training=True)
            loss = loss_fn(labels, preds)
            acc = accuracy_metric(labels, preds)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, acc

    EPOCHS = cfg.epochs
    best_acc = 0.0
    best_model_path = os.path.join(checkpoint_dir, "best_model.weights.h5")
    optimizer = tf.keras.optimizers.AdamW(learning_rate=cfg.lr, weight_decay=cfg.weight_decay)
    
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0

        for step in range(len(train_dataset)):
            inputs = train_dataset[step][0]
            labels = train_dataset[step][1]
            loss, acc = train_step(inputs, labels)
            total_loss += loss.numpy()
            total_acc += acc.numpy()
            num_batches += 1

            print(f"Step {step}: Loss = {loss.numpy():.4f}, Accuracy = {acc.numpy():.4f}")

        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_acc:.4f}")

        # Ghi thông tin log vào file
        logging.info(f"Epoch {epoch+1}/{EPOCHS} - Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_acc:.4f}")
        
        test_loss,test_acc = evaluate(model, test_dataset)
        print(f"Test Set - Avg Loss: {test_loss:.4f}, Avg Accuracy: {test_acc:.4f}")
        logging.info(f"Test Set - Avg Loss: {test_loss:.4f}, Avg Accuracy: {test_acc:.4f}")

        checkpoint.save(file_prefix=checkpoint_prefix)  
        clean_old_checkpoints(checkpoint_dir=checkpoint_dir, max_to_keep=3)

        train_dataset.on_epoch_end()
        if avg_acc >= best_acc:
            model.save_weights(best_model_path)
            best_acc = avg_acc
        
            
