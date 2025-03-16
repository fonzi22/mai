import argparse
import os
import yaml
import tensorflow as tf


def get_config():
    parser = argparse.ArgumentParser(description="ReMixMatch TensorFlow")
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('--save_name', type=str, default='remixmatch')
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint", default=False  )
    parser.add_argument("--resume_model_only", action="store_true", help="Resume only model weights", default=False)
    parser.add_argument("--load_path", type=str, help="Path to load model checkpoint")
    parser.add_argument("--overwrite", action="store_true", default=True)
    parser.add_argument("--config_file", type=str, default="", help="Path to config file")


    # train para
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--uratio", type=int, default=7,
                        help="Ratio of unlabeled to labeled data in each batch")
    parser.add_argument("--num_workers", type=int, default=4)


    # optimize conf
    parser.add_argument("--optimizer", type=str, default="Adam",
                        choices=["SGD", "Adam", "AdamW", "RMSprop"])
    parser.add_argument("--lr", type=float, default=0.002, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--nesterov", action="store_true", default=False, help="Use Nesterov momentum")
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "step", "exponential", "none"])
    parser.add_argument("--warmup_epochs", type=int, default=10, help="LR warmup epochs")

    
    # dataset
    parser.add_argument("--dataset", type=str, default="custom",
                        choices=["cifar10", "cifar100", "svhn", "stl10", "custom"])

    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--labeled_data_dir", type=str, default=None,
                        help="Directory containing labeled data (organized in class folders)")
    parser.add_argument("--unlabeled_data_dir", type=str, default=None,
                        help="Directory containing unlabeled data (can be flat structure)")

        
    parser.add_argument("--val_data_dir", type=str, default=None,
                        help="Directory containing validation data (organized in class folders)")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--img_size", type=int, default=32)


    # model
    parser.add_argument("--model", type=str, default="mobilenetv3",
                        choices=["efficientnetv2", "mobilenetv3", "resnet101"])

    
    # remixmatch specific config
    parser.add_argument("--T", type=float, default=0.5,help="Temperature for sharpening")
    parser.add_argument("--kl_loss_ratio", type=float, default=0.5,help="Weight for KL loss")
    parser.add_argument("--rot_loss_ratio", type=float, default=0.5, help="Weight for rotation loss")
    parser.add_argument("--mixup_alpha", type=float, default=0.75, help="Alpha parameter for Beta distribution in MixUp")
    parser.add_argument("--mixup_manifold", action="store_true", default=False, help="Use manifold mixup")
    parser.add_argument("--unsup_warm_up", type=float, default=0.015625, help="Warm-up coefficient for unsupervised loss (1/64)")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="Decay rate for EMA model")


    # other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log_interval", type=int, default=10,help="Number of iterations between logging")
    parser.add_argument("--eval_interval", type=int, default=1,help="Number of epochs between evaluation")
    parser.add_argument("--gpu", type=str, default="0", help="GPU index to use")


    # wandb
    args = parser.parse_args()

    if args.config_file:
        with open(args.config_file, 'r') as f:
            config = yaml.safe_load(f)
            for key, value in config.items():
                if key in vars(args):
                    setattr(args, key, value)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    save_path = os.path.join(args.save_dir, args.save_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    elif os.path.exists(save_path) and args.overwrite and not args.resume:
        import shutil
        shutil.rmtree(save_path)
        os.makedirs(save_path, exist_ok=True)
    
    tf.random.set_seed(args.seed)
    return args

