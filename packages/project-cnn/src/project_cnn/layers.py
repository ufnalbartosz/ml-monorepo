"""Custom Keras layers needed by the ported models.

Keras 3 has no local-response-normalization layer - it fell out of fashion once
batch normalization arrived - but both AlexNet and the inception model in this
package use it, so it is reimplemented here on top of the TensorFlow op.
"""

from __future__ import annotations

import keras
import tensorflow as tf


@keras.saving.register_keras_serializable(package="project_cnn")
class LocalResponseNormalization(keras.layers.Layer):
    """Across-channel local response normalization, as in the AlexNet paper.

    The defaults match ``tflearn.layers.normalization.local_response_normalization``,
    which is what the pre-port models used.
    """

    def __init__(
        self,
        depth_radius: int = 5,
        bias: float = 1.0,
        alpha: float = 0.0001,
        beta: float = 0.75,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.depth_radius = depth_radius
        self.bias = bias
        self.alpha = alpha
        self.beta = beta

    def call(self, inputs):
        return tf.nn.local_response_normalization(
            inputs,
            depth_radius=self.depth_radius,
            bias=self.bias,
            alpha=self.alpha,
            beta=self.beta,
        )

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            depth_radius=self.depth_radius,
            bias=self.bias,
            alpha=self.alpha,
            beta=self.beta,
        )
        return config
