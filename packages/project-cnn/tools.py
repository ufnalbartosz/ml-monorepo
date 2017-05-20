import tensorflow as tf


def pre_process_image(image, training, img_size_cropped, num_channels):
    # This function takes a single image as input,
    # and a boolean whether to build the training or testing graph.

    if training:
        # For training, add the following to the TensorFlow graph.
        # Randomly crop the input image.
        image = tf.random_crop(
            image, size=[img_size_cropped, img_size_cropped, num_channels])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)

        # Randomly adjust hue, contrast and saturation.
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

        # Some of these functions may overflow and result in pixel
        # values beyond the [0, 1] range. It is unclear from the
        # documentation of TensorFlow 0.10.0rc0 whether this is
        # intended. A simple solution is to limit the range.

        # Limit the image pixels between [0, 1] in case of overflow.
        image = tf.minimum(image, 1.0)
        image = tf.maximum(image, 0.0)
    else:
        # For training, add the following to the TensorFlow graph.
        # Crop the input image around the centre so it is the same
        # size as images that are randomly cropped during training.
        image = tf.image.resize_image_with_crop_or_pad(
            image,
            target_height=img_size_cropped,
            target_width=img_size_cropped
        )

    return image


def pre_process(images, training, img_size_cropped, num_channels):
    # Use TensorFlow to loop over all the input images and call
    # the function above which takes a single image as input.
    images = tf.map_fn(
        lambda image: pre_process_image(image, training, img_size_cropped,
                                        num_channels),
        images)

    return images


def get_weights_variable(layer_name):
    # Retrieve an existing variable named 'weights' in the scope
    # with the given layer_name.
    # This is awkward because the TensorFlow function was
    # really intended for another purpose.

    with tf.variable_scope("network/" + layer_name, reuse=True):
        variable = tf.get_variable('weights')

    return variable


def get_layer_output(layer_name):
    # The name of the last operation of the convolutional layer.
    # This assumes you are using Relu as the activation-function.
    tensor_name = "network/" + layer_name + "/Relu:0"

    # Get the tensor with this name.
    tensor = tf.get_default_graph().get_tensor_by_name(tensor_name)

    return tensor




#####################################
# examples
# line 93 main.py
# weights_conv1 = tools.get_weights_variable(layer_name='layer_conv1')
# weights_conv2 = tools.get_weights_variable(layer_name='layer_conv2')

# output_conv1 = get_layer_output(layer_name='layer_conv1')
# output_conv2 = get_layer_output(layer_name='layer_conv2')

# line 293 main.py
# def get_test_image(i):
#     return images_test[i, :, :, :], cls_test[i]

# img, cls = get_test_image(16)
# plot.plot_distorted_image(img, cls)

# line 300 main.py
# plot_conv_weights(weights=weights_conv1, input_channel=0)
# plot_conv_weights(weights=weights_conv2, input_channel=1)

# img, cls = get_test_image(16)
# plot.plot_image(img)

# plot_layer_output(output_conv1, image=img)
# plot_layer_output(output_conv2, image=img)

# label_pred, cls_pred = session.run([y_pred, y_pred_cls],
#                                    feed_dict={x: [img]})

# line 305 main.py
# Print the predicted label.
# print(label_pred[0])
# class_names[3]
# class_names[5]
