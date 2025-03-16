# import tensorflow as tf
# from models import MobilenetV3, Resnet101
# from utils import *
# import argparse
# import configs
# from tensorflow.keras.mixed_precision import set_global_policy

# set_global_policy('mixed_float16')


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model", required=True, type=str,
#                         help="Model name: [resnet101, mobilenetv3]")
    
#     args = parser.parse_args()
#     if args.model == "resnet101":
#         import configs.resnet101_config as cfg
#         model = Resnet101(cfg.num_classes, cfg.url)
#     elif args.model == "mobilenetv3":
#         import configs.mobilenetv3_config as cfg
#         model = MobilenetV3(cfg.num_classes, (cfg.crop_size, cfg.crop_size, 3))
        
#     if cfg.backbone_trainable:
#         model.backbone.trainable = True
#     else:
#         model.backbone.trainable = False
        
#     train_dataset = get_dataset(cfg, cfg.train_folder, mode='train')
#     print(f"Batch size: {train_dataset.batch_size}")


#     lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(cfg.boundaries, cfg.lr)
#     optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=cfg.weight_decay)

#     model.compile(
#         optimizer=optimizer,
#         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
#         metrics=['accuracy']
#     )
    
#     checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#         filepath=os.path.join(cfg.save_path, "epoch_{epoch:02d}.h5"),  # Save each epoch separately
#         save_weights_only=False,  # Save full model (architecture + weights)
#         save_freq="epoch",  # Save at the end of each epoch
#     )

    
#     model.fit(
#         train_dataset,
#         epochs=cfg.epochs,              
#         steps_per_epoch=len(train_dataset),
#         callbacks=[checkpoint_callback],
#         verbose=True
#     )

import os
import tensorflow as tf
from models import MobilenetV3, Resnet101
from utils import *
import argparse
import configs
from tensorflow.keras.mixed_precision import set_global_policy, LossScaleOptimizer

# ✅ Enable GPU Memory Growth to Prevent OOM Errors
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# ✅ Enable Mixed Precision Training
set_global_policy('mixed_float16')

# ✅ Enable XLA Compilation for Speed
tf.config.optimizer.set_jit(True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str,
                        help="Model name: [resnet101, mobilenetv3]")
    
    args = parser.parse_args()
    if args.model == "resnet101":
        import configs.resnet101_config as cfg
        model = Resnet101(cfg.num_classes, cfg.url)
    elif args.model == "mobilenetv3":
        import configs.mobilenetv3_config as cfg
        model = MobilenetV3(cfg.num_classes, (cfg.crop_size, cfg.crop_size, 3))
        
    model.backbone.trainable = cfg.backbone_trainable

    # ✅ Optimize Dataset Pipeline
    train_dataset = get_dataset(cfg, cfg.train_folder, mode='train')

    # # ✅ Print Batch Size Correctly
    # for batch in train_dataset.take(1):
    #     print(f"Batch size: {batch[0].shape[0]}")

    # ✅ Use LossScaleOptimizer for Mixed Precision
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(cfg.boundaries, cfg.lr)
    optimizer = LossScaleOptimizer(
        tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=cfg.weight_decay)
    )

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    # ✅ Optimize Model Checkpointing (Only Save Weights)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(cfg.save_path, "epoch_{epoch:02d}.weights.h5"),
        save_weights_only=True,  # ✅ Save only weights to reduce memory
        save_freq="epoch",
    )

    # ✅ Train Model
    model.fit(
        train_dataset,
        epochs=cfg.epochs,
        steps_per_epoch=len(train_dataset),
        callbacks=[checkpoint_callback],
        verbose=True
    )
