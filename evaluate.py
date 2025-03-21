import tensorflow as tf
import logging
from tqdm import tqdm

def evaluate(model, test_dataset):
    """
    Đánh giá mô hình trên tập test dataset.
    
    Args:
        model (tf.keras.Model): Mô hình đã huấn luyện.
        test_dataset (tf.data.Dataset): Tập dữ liệu kiểm tra.
    """
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    
    for step in tqdm(range(len(test_dataset))):
        batch = test_dataset[step]
        inputs = batch[0]
        labels = batch[1]
        preds = model(inputs, training=False)
        loss = loss_fn(labels, preds)
        acc = accuracy_metric(labels, preds)
        
        total_loss += loss.numpy()
        total_acc += acc.numpy()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    
    return avg_loss, avg_acc