import tensorflow as tf
import tensorflow_hub as hub

class Resnet101(tf.keras.Model):
    def __init__(self, num_classes, url, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.url = url
        self.head = tf.keras.layers.Dense(num_classes, activation='softmax')
        self.backbone = hub.KerasLayer(url, trainable=False)

    def call(self, inputs):
        embedding = self.backbone(inputs)
        return self.head(embedding)

    def build(self, input_shape):
        self.backbone.build(input_shape)
        backbone_output_shape = self.backbone.compute_output_shape(input_shape)
        self.head.build(backbone_output_shape)
        super().build(input_shape)