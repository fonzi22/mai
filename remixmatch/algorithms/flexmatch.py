import tensorflow as tf
from ..data import weak_augment, strong_augment

@tf.function
def flexmatch_train_step(model, optimizer, x_lb, y_lb, x_ulb, class_thresholds, lambda_u=1.0, T=0.5):
    """
    FlexMatch training step:
      - Labeled branch: tính supervised loss.
      - Unlabeled branch: sử dụng weak augmentation để tạo pseudo-label,
        sau đó so sánh max probability với threshold tương ứng của lớp đó.
      - Nếu vượt ngưỡng, tính consistency loss giữa strong augmented branch và pseudo-label.
    """
    # Tạo phiên bản weak và strong cho unlabeled images
    x_ulb_w = tf.map_fn(weak_augment, x_ulb)
    x_ulb_s = tf.map_fn(strong_augment, x_ulb)
    
    with tf.GradientTape() as tape:
        # Labeled data forward
        logits_lb = model(x_lb, training=True)
        sup_loss = tf.keras.losses.sparse_categorical_crossentropy(y_lb, logits_lb, from_logits=True)
        sup_loss = tf.reduce_mean(sup_loss)
        
        # Unlabeled weak branch: tính pseudo-label và mask theo ngưỡng động
        logits_ulb_w = model(x_ulb_w, training=True)
        probs_ulb_w = tf.nn.softmax(logits_ulb_w, axis=-1)
        max_probs = tf.reduce_max(probs_ulb_w, axis=-1)
        pseudo_labels = tf.argmax(probs_ulb_w, axis=-1, output_type=tf.int32)
        # Lấy ngưỡng tương ứng cho mỗi mẫu theo pseudo_label
        sample_thresholds = tf.gather(class_thresholds, pseudo_labels)
        mask = tf.cast(max_probs >= sample_thresholds, tf.float32)
        
        # Unlabeled strong branch
        logits_ulb_s = model(x_ulb_s, training=True)
        unsup_loss = tf.keras.losses.sparse_categorical_crossentropy(pseudo_labels, logits_ulb_s, from_logits=True)
        unsup_loss = tf.reduce_mean(unsup_loss * mask)
        
        total_loss = sup_loss + lambda_u * unsup_loss
        
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return sup_loss, unsup_loss, total_loss, tf.reduce_mean(mask)