"""Image pre-processing and model introspection, ported to TF2.

The TF1 version of this module leaned on graph-mode globals: ``pre_process``
was called twice to build two copies of the graph under a
``tf.variable_scope(reuse=...)``, and the introspection helpers reached into
``tf.get_default_graph()`` by string name.  Neither exists in TF2.

Here, pre-processing is an ordinary function of a tensor plus a
:class:`PreProcessing` layer that reads Keras' own ``training`` flag, so one
model covers both the distorted training path and the deterministic evaluation
path.  Introspection goes through the ``keras.Model`` object instead of the
global graph.
"""

from __future__ import annotations

import keras
import tensorflow as tf


def pre_process_image(
    image: tf.Tensor,
    training: bool,
    img_size_cropped: int,
    num_channels: int,
) -> tf.Tensor:
    """Distort one image for training, or centre-crop it for evaluation.

    During training the image is randomly cropped, flipped and colour-jittered.
    During evaluation it is only cropped around the centre, so that accuracy is
    measured on a deterministic input and is reproducible between runs.
    """
    if training:
        image = tf.image.random_crop(image, size=[img_size_cropped, img_size_cropped, num_channels])
        image = tf.image.random_flip_left_right(image)

        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

        # The colour ops can push pixels outside [0, 1]; put them back.
        image = tf.clip_by_value(image, 0.0, 1.0)
    else:
        image = tf.image.resize_with_crop_or_pad(
            image,
            target_height=img_size_cropped,
            target_width=img_size_cropped,
        )

    return image


def pre_process(
    images: tf.Tensor,
    training: bool,
    img_size_cropped: int,
    num_channels: int,
) -> tf.Tensor:
    """Apply :func:`pre_process_image` to every image in a batch.

    ``map_fn`` rather than a batched op, because each image must get its own
    random crop and colour jitter.
    """
    return tf.map_fn(
        lambda image: pre_process_image(image, training, img_size_cropped, num_channels),
        images,
    )


@keras.saving.register_keras_serializable(package="project_cnn")
class PreProcessing(keras.layers.Layer):
    """Keras layer wrapping :func:`pre_process`.

    Keras passes ``training`` down through ``__call__``, which replaces the TF1
    trick of building the graph twice under a reusing variable scope.
    """

    def __init__(self, img_size_cropped: int = 24, num_channels: int = 3, **kwargs) -> None:
        super().__init__(**kwargs)
        self.img_size_cropped = img_size_cropped
        self.num_channels = num_channels

    def call(self, inputs, training=None):
        if training:
            return pre_process(inputs, True, self.img_size_cropped, self.num_channels)
        return pre_process(inputs, False, self.img_size_cropped, self.num_channels)

    def compute_output_shape(self, input_shape):
        batch, *_ = input_shape
        return (batch, self.img_size_cropped, self.img_size_cropped, self.num_channels)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            img_size_cropped=self.img_size_cropped,
            num_channels=self.num_channels,
        )
        return config


def get_weights_variable(model: keras.Model, layer_name: str):
    """Return the kernel of a convolutional or dense layer by name.

    The TF1 version had to re-enter ``tf.variable_scope("network/" + name,
    reuse=True)`` and call ``tf.get_variable('weights')``.  A Keras model just
    knows its layers.
    """
    layer = model.get_layer(layer_name)

    if not hasattr(layer, "kernel"):
        raise ValueError(f"Layer {layer_name!r} has no kernel (type {type(layer).__name__})")

    return layer.kernel


def get_layer_output(model: keras.Model, layer_name: str) -> keras.Model:
    """Return a model that maps the original input to ``layer_name``'s output.

    Replaces the TF1 ``graph.get_tensor_by_name("network/<name>/Relu:0")``
    lookup, which silently depended on ReLU being the activation.
    """
    layer = model.get_layer(layer_name)

    return keras.Model(inputs=model.inputs, outputs=layer.output, name=f"{layer_name}_output")
