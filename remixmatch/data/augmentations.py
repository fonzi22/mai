import tensorflow as tf
import albumentations as A


def weak_augment(size: int, padding: int):
    """
    Create weak augmentation
    """
    return A.Compose(
        [
            A.PadIfNeeded(min_height=size + 2 * padding, min_width=size + 2 * padding, border_mode=0),
            A.RandomCrop(height=size, width=size),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=0, std=1, max_pixel_value=255.0),
        ],
        p=1.0
    )



def strong_augment(size: int):
    
    return A.Compose(
        [
            A.PadIfNeeded(min_height=size + 8, min_width=size + 8, border_mode=0),
            A.RandomCrop(height=size, width=size),
            A.HorizontalFlip(p=0.5),
            #strong augmentation
            A.OneOf([
                A.RandomBrightness(limit=0.8, p=0.5),
                A.RandomContrast(limit=(0.2, 1.8), p=0.5),
                A.RandomSaturation(limit=(0.2, 1.8), p=0.5),
                A.RandomHue(hue_limit=0.2, p=0.5),
            ], p=0.8),
            A.CoarseDropout(max_holes=1, max_height=size//4, max_width=size//4, fill_value=0, p=0.5),
        ],
        p=1.0
    )


def rotation_augment():
    return A.Compose(
        [
            A.RandomRotate90(p=1.0, always_apply=True)
        ],
        p=1.0
    )

def mixup(x1, x2, y1, y2, alpha=0.75):
    
    batch_size = tf.shape(x1)[0]
    beta_dist = tf.compat.v1.distributions.Beta(alpha, alpha)
    lam = beta_dist.sample(batch_size)

    lam_x = tf.reshape(lam, [batch_size, 1, 1, 1])
    lam_y = tf.reshape(lam, [batch_size, 1])

    mixed_x = lam_x * x1 + (1 - lam_x) * x2
    mixed_y = lam_y * y1 + (1 - lam_y) * y2

    return mixed_x, mixed_y, lam


