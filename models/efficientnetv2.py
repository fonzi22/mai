import tensorflow as tf

class EfficientNetV2(tf.keras.Model):
    def __init__(self, num_classes, input_shape=(128, 128, 3)):
        super().__init__()

        # Load EfficientNet with ImageNet weights, without the top classifier
        self.backbone = tf.keras.applications.EfficientNetV2L(
            input_shape=input_shape,
            weights='imagenet',
            include_top=False,
            pooling='avg', 
        )

        self.head = tf.keras.layers.Dense(num_classes,  activation='softmax')
    
    def call(self, images):
        embedding = self.backbone(images)
        return self.head(embedding)

# model = EfficientNetV2(num_classes=5)
# x = tf.random.normal([1, 128, 128, 3])

# # Forward pass
# logits = model(x)

# print("Logits shape:", logits.shape)