import tensorflow as tf
from ..data import weak_augment, strong_augment

@tf.function
def fixmatch_train_step(model, optimizer, x_lb, y_lb, x_ulb, p_cutoff=0.95, lambda_u=1.0, T=0.5):
    """
    Thực hiện một bước train theo FixMatch:
      - Tính supervised loss từ labeled data.
      - Với unlabeled data: tạo weak và strong augmentations.
      - Tính pseudo-label từ weak branch nếu độ tin cậy (max probability) vượt p_cutoff.
      - Tính unsupervised consistency loss (cross-entropy) giữa pseudo-label và strong branch, chỉ trên các mẫu đủ tin cậy.
    """
    # Tạo phiên bản weak và strong cho unlabeled images
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
        
        # (Tùy chọn) Có thể áp dụng sharpening bằng nhiệt độ T vào probs_ulb_w trước khi lấy pseudo-label
        
        # Unlabeled: strong branch để tính consistency loss
        logits_ulb_s = model(x_ulb_s, training=True)
        unsup_loss = tf.keras.losses.sparse_categorical_crossentropy(pseudo_labels, logits_ulb_s, from_logits=True)
        unsup_loss = tf.reduce_mean(unsup_loss * mask)
        
        total_loss = sup_loss + lambda_u * unsup_loss
    
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return sup_loss, unsup_loss, total_loss, tf.reduce_mean(mask)