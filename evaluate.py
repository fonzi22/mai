# ✅ Reload the best model weights
model.load_weights(os.path.join(cfg.save_path, "epoch_best.weights.h5"))

# ✅ Evaluate on validation set
val_loss, val_acc = model.evaluate(val_dataset, verbose=1)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")

# ✅ Evaluate on test set (if available)
test_loss, test_acc = model.evaluate(test_dataset, verbose=1)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
