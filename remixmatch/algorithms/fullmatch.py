import tensorflow as tf
from data import weak_augment, strong_augment

@tf.function
def fullmatch_train_step(model, optimizer, x_lb, y_lb, x_ulb, p_cutoff=0.95, lambda_u=1.0, lambda_p=0.5, T=0.5):
    """
    FullMatch training step:
      - Tính supervised loss trên dữ liệu labeled.
      - Với dữ liệu unlabeled:
          + Tạo 2 phiên bản: weak (để tạo pseudo-label) và strong (để tính consistency loss).
          + Tính pseudo-label từ weak branch nếu độ tin cậy (max probability) vượt ngưỡng p_cutoff.
          + Tính unsupervised loss (cross-entropy) giữa pseudo-label và dự đoán strong branch (chỉ với các mẫu đủ tin cậy).
          + Tính penalty loss dựa trên entropy của phân phối dự đoán (giảm entropy để ép model đưa ra dự đoán chắc chắn hơn).
      - Tổng loss = L_sup + lambda_u * L_unsup + lambda_p * L_penalty.
    """
    # Tạo weak và strong augmentations cho dữ liệu unlabeled
    x_ulb_w = tf.map_fn(weak_augment, x_ulb)
    x_ulb_s = tf.map_fn(strong_augment, x_ulb)
    
    with tf.GradientTape() as tape:
        # Labeled forward pass
        logits_lb = model(x_lb, training=True)
        sup_loss = tf.keras.losses.sparse_categorical_crossentropy(y_lb, logits_lb, from_logits=True)
        sup_loss = tf.reduce_mean(sup_loss)
        
        # Unlabeled: weak branch để tạo pseudo-label
        logits_ulb_w = model(x_ulb_w, training=True)
        probs_ulb_w = tf.nn.softmax(logits_ulb_w, axis=-1)
        max_probs = tf.reduce_max(probs_ulb_w, axis=-1)
        pseudo_labels = tf.argmax(probs_ulb_w, axis=-1, output_type=tf.int32)
        mask = tf.cast(max_probs >= p_cutoff, tf.float32)
        
        # Unlabeled: strong branch để tính consistency loss
        logits_ulb_s = model(x_ulb_s, training=True)
        unsup_loss = tf.keras.losses.sparse_categorical_crossentropy(pseudo_labels, logits_ulb_s, from_logits=True)
        unsup_loss = tf.reduce_mean(unsup_loss * mask)
        
        # Penalty loss: entropy minimization trên phân phối của weak branch
        entropy = - tf.reduce_sum(probs_ulb_w * tf.math.log(probs_ulb_w + 1e-8), axis=-1)
        penalty_loss = tf.reduce_mean(entropy)
        
        total_loss = sup_loss + lambda_u * unsup_loss + lambda_p * penalty_loss
        
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return sup_loss, unsup_loss, total_loss, tf.reduce_mean(mask)