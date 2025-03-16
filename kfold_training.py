import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=21000)]  # Giới hạn 4GB
)

from models import MobilenetV3, Resnet101, EfficientNetV2
from utils import *
import argparse
import configs
from sklearn.model_selection import KFold



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str,
                        help="Model name: [resnet101, mobilenetv3, efficientnetv2]")
    args = parser.parse_args()

    # Import cấu hình tương ứng
    if args.model == "resnet101":
        import configs.resnet101_config as cfg
    elif args.model == "efficientnetv2":
        import configs.efficientnetv2_config as cfg
    elif args.model == "mobilenetv3":
        import configs.mobilenetv3_config as cfg
    else:
        raise ValueError("Model not supported!")

    # Số fold cho cross validation (ví dụ: 5-fold)
    num_folds = cfg.num_folds if hasattr(cfg, "num_folds") else 5

    # Load toàn bộ data từ folder training
    all_image_paths, all_labels = load_data(cfg, cfg.train_folder)
    # Khởi tạo KFold
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_no = 1

    for train_idx, val_idx in kfold.split(all_image_paths):
        print(f"\n================ Fold {fold_no} ================\n")

        # Tách train và validation cho fold hiện tại
        train_paths = [all_image_paths[i] for i in train_idx]
        train_labels = [all_labels[i] for i in train_idx]
        val_paths = [all_image_paths[i] for i in val_idx]
        val_labels = [all_labels[i] for i in val_idx]

        # Tạo dataset cho train và validation
        train_dataset = Dataset(
            image_paths=train_paths,
            labels=train_labels,
            batch_size=cfg.batch_size,
            resize_size=cfg.resize_size,
            crop_size=cfg.crop_size,
            num_classes=cfg.num_classes,
            shuffle=True
        )

        val_dataset = Dataset(
            image_paths=val_paths,
            labels=val_labels,
            batch_size=cfg.batch_size,
            resize_size=cfg.resize_size,
            crop_size=cfg.crop_size,
            num_classes=cfg.num_classes,
            shuffle=False
        )

        # Khởi tạo model mới cho mỗi fold
        if args.model == "resnet101":
            model = Resnet101(cfg.num_classes, cfg.url)
        elif args.model == "efficientnetv2":
            model = EfficientNetV2(cfg.num_classes, (cfg.crop_size, cfg.crop_size, 3))
        elif args.model == "mobilenetv3":
            model = MobilenetV3(cfg.num_classes, (cfg.crop_size, cfg.crop_size, 3))
        
        # Nếu không muốn train backbone
        model.backbone.trainable = cfg.backbone_trainable

        # Khởi tạo optimizer, loss và các metric
        if isinstance(cfg.lr, list):
            lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(cfg.boundaries, cfg.lr)
            optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=cfg.weight_decay)
        else:
            optimizer = tf.keras.optimizers.AdamW(learning_rate=cfg.lr, weight_decay=cfg.weight_decay)

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        
        @tf.function
        def train_step(inputs, labels):
            with tf.GradientTape() as tape:
                preds = model(inputs, training=True)
                loss = loss_fn(labels, preds)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            train_acc_metric.update_state(labels, preds)
            return loss

        @tf.function
        def val_step(inputs, labels):
            preds = model(inputs, training=False)
            loss = loss_fn(labels, preds)
            val_acc_metric.update_state(labels, preds)
            return loss
        
        best_val_acc = 0.0
        for epoch in range(cfg.epochs):
            print(f"Fold {fold_no} - Epoch {epoch+1}/{cfg.epochs}")
            train_acc_metric.reset_state()
            val_acc_metric.reset_state()
            total_train_loss = 0.0
            num_train_batches = 0

            for step in range(len(train_dataset)):
                inputs = train_dataset[step][0]
                labels = train_dataset[step][1]
                loss = train_step(inputs, labels)
                total_train_loss += loss.numpy()
                num_train_batches += 1
                print(f"Step {step}: Loss = {loss.numpy():.4f}, Training Accuracy = {train_acc_metric.result().numpy():.4f}")

            avg_train_loss = total_train_loss / num_train_batches
            avg_train_acc = train_acc_metric.result().numpy()
            print(f"Fold {fold_no} Epoch {epoch+1}: Avg Training Loss = {avg_train_loss:.4f}, Avg Training Accuracy = {avg_train_acc:.4f}")

            # Validation
            total_val_loss = 0.0
            num_val_batches = 0
            for step in range(len(val_dataset)):
                inputs = val_dataset[step][0]
                labels = val_dataset[step][1]
                loss = val_step(inputs, labels)
                total_val_loss += loss.numpy()
                num_val_batches += 1

            avg_val_loss = total_val_loss / num_val_batches
            avg_val_acc = val_acc_metric.result().numpy()
            print(f"Fold {fold_no} Epoch {epoch+1}: Avg Validation Loss = {avg_val_loss:.4f}, Avg Validation Accuracy = {avg_val_acc:.4f}")

            # Lưu model tốt nhất dựa trên validation accuracy
            if avg_val_acc > best_val_acc:
                best_val_acc = avg_val_acc
                fold_best_path = os.path.join(cfg.save_path, args.model, f"fold_{fold_no}_best_model.h5")
                os.makedirs(os.path.dirname(fold_best_path), exist_ok=True)
                model.save(fold_best_path)
                print(f"Fold {fold_no}: Best model updated with Val Accuracy = {avg_val_acc:.4f}")

        fold_no += 1
        tf.keras.backend.clear_session()
