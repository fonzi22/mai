import tensorflow as tf
from ..data import weak_augment, strong_augment

@tf.function
def mixmatch_train_step(model, optimizer, x_lb, y_lb, x_ulb, K=2, T=0.5, alpha=0.75, lambda_u=1.0):
    """
    MixMatch training step:
      1. Với unlabeled data: áp dụng K lần weak augmentation, lấy trung bình các dự đoán,
         sau đó sharpen (nâng cao độ sắc) guessed label.
      2. Gộp labeled data (one-hot label) với unlabeled data (guessed label) và thực hiện mixup.
      3. Tính supervised loss trên dữ liệu labeled và consistency loss trên dữ liệu unlabeled đã mixup.
    """
    num_classes = model.output_shape[-1]
    
    # Augment unlabeled data K lần và tính trung bình dự đoán
    preds_ulb = []
    for _ in range(K):
        aug = tf.map_fn(weak_augment, x_ulb)
        logits = model(aug, training=True)
        preds_ulb.append(tf.nn.softmax(logits, axis=-1))
    preds_ulb = tf.stack(preds_ulb, axis=0)
    avg_preds = tf.reduce_mean(preds_ulb, axis=0)
    
    # Sharpening: dùng nhiệt độ T
    def sharpen(p, T):
        p_power = tf.pow(p, 1.0 / T)
        return p_power / tf.reduce_sum(p_power, axis=-1, keepdims=True)
    guessed_labels = sharpen(avg_preds, T)
    
    # Chuẩn bị one-hot cho labeled data
    y_lb_oh = tf.one_hot(y_lb, depth=num_classes)
    
    # Kết hợp labeled và unlabeled data
    x_combined = tf.concat([x_lb, x_ulb], axis=0)
    y_combined = tf.concat([y_lb_oh, guessed_labels], axis=0)
    
    # MixUp: trộn ngẫu nhiên theo hệ số lambda
    batch_size = tf.shape(x_combined)[0]
    # Lấy lambda từ phân phối Beta
    lam = tf.compat.v1.distributions.Beta(alpha, alpha).sample([batch_size])
    lam = tf.maximum(lam, 1 - lam)  # theo công thức MixUp
    lam = tf.reshape(lam, (batch_size, 1, 1, 1))
    
    # Shuffle các cặp dữ liệu
    indices = tf.random.shuffle(tf.range(batch_size))
    x_shuffled = tf.gather(x_combined, indices)
    y_shuffled = tf.gather(y_combined, indices)
    
    x_mixed = lam * x_combined + (1 - lam) * x_shuffled
    # Với label, reshape lam cho phù hợp với vector
    lam_label = tf.reshape(lam, (batch_size, 1))
    y_mixed = lam_label * y_combined + (1 - lam_label) * y_shuffled
    
    with tf.GradientTape() as tape:
        logits = model(x_mixed, training=True)
        # Tính supervised loss cho phần labeled
        loss_sup = tf.keras.losses.categorical_crossentropy(y_mixed[:tf.shape(x_lb)[0]], logits[:tf.shape(x_lb)[0]], from_logits=True)
        loss_sup = tf.reduce_mean(loss_sup)
        # Tính unsupervised loss cho phần unlabeled (cho cả dữ liệu trộn)
        loss_unsup = tf.keras.losses.categorical_crossentropy(y_mixed, logits, from_logits=True)
        loss_unsup = tf.reduce_mean(loss_unsup)
        total_loss = loss_sup + lambda_u * loss_unsup
        
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss_sup, loss_unsup, total_loss, None