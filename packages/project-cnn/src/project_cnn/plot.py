import math
import matplotlib
# Select a non-interactive backend before pyplot is imported, because the
# training runs on a headless machine where plt.show() has nothing to
# draw on. The plots are written to files instead.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix


def plot_images(images, cls_true, class_names, cls_pred=None, smooth=True,
                filename=None):
    assert len(images) == len(cls_true)

    # Plot at most 9 images, but do not require exactly 9 of them - there
    # may be fewer misclassified images than that.
    num_images = min(len(images), 9)
    if num_images == 0:
        print("Nothing to plot.")
        return

    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)

    # Adjust vertical spacing if we need to print ensemble and best-net.
    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        # Blank out the sub-plots we have no image for.
        if i >= num_images:
            ax.axis('off')
            continue

        # Interpolation type.
        if smooth:
            interpolation = 'spline16'
        else:
            interpolation = 'nearest'

        # Plot image.
        ax.imshow(images[i, :, :, :], interpolation=interpolation)

        # Name of the true class.
        cls_true_name = class_names[cls_true[i]]

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true_name)
        else:
            # Name of the predicted class.
            cls_pred_name = class_names[cls_pred[i]]

            xlabel = "True: {0}\nPred: {1}".format(
                cls_true_name, cls_pred_name)

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Write the figure to disk, otherwise all of the work above is
    # thrown away without anything being displayed.
    if filename is not None:
        fig.savefig(filename, bbox_inches='tight')
        print("Saved plot to: " + filename)

    # Release the figure, otherwise matplotlib keeps every figure alive
    # and the memory usage grows with each call.
    plt.close(fig)


def plot_example_errors(cls_pred, correct, dataset):
    # This function is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # correct is a boolean array whether the predicted class
    # is equal to the true class for each image in the test-set.

    # Negate the boolean array.
    incorrect = (correct == False)
    # Get the images from the test-set that have been
    # incorrectly classified.
    images_test = dataset['test_images']
    images = images_test[incorrect]

    if len(images) == 0:
        print("No misclassified images to plot.")
        return

    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_test = dataset['test_cls']
    cls_true = cls_test[incorrect]
    # Plot the first 9 images.
    class_names = dataset['class_names']
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9],
                class_names=class_names,
                filename='example_errors.png')


def plot_confusion_matrix(cls_pred, dataset):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the confusion matrix using sklearn.
    cls_test = dataset['test_cls']
    cm = confusion_matrix(y_true=cls_test,  # True class for test-set.
                          y_pred=cls_pred)  # Predicted class.

    # Print the confusion matrix as text.
    class_names = dataset['class_names']
    num_classes = len(class_names)
    for i in range(num_classes):
        # Append the class-name to each line.
        row = ''.join('%04s ' % x for x in cm[i])

        class_name = " ({:>2}) {}".format(i, class_names[i])
        print(row + class_name)

    # Print the class-numbers for easy reference.
    class_numbers = [" (%02s)" % i for i in range(num_classes)]
    print("".join(class_numbers))


def plot_conv_weights(weights, session, input_channel=0, filename='conv_weights.png'):
    # Assume weights are TensorFlow ops for 4-dim variables
    # e.g. weights_conv1 or weights_conv2.

    # Retrieve the values of the weight-variables from TensorFlow.
    # A feed-dict is not necessary because nothing is calculated.
    w = session.run(weights)

    # Print statistics for the weights.
    print("Min:  {0:.5f}, Max:   {1:.5f}".format(w.min(), w.max()))
    print("Mean: {0:.5f}, Stdev: {1:.5f}".format(w.mean(), w.std()))

    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)
    abs_max = max(abs(w_min), abs(w_max))

    # Number of filters used in the conv. layer.
    num_filters = w.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = int(math.ceil(math.sqrt(num_filters)))

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i < num_filters:
            # Get the weights for the i'th filter of the input channel.
            # The format of this 4-dim tensor is determined by the
            # TensorFlow API. See Tutorial #02 for more details.
            img = w[:, :, input_channel, i]

            # Plot image.
            ax.imshow(img, vmin=-abs_max, vmax=abs_max,
                      interpolation='nearest', cmap='seismic')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Write the figure to disk, otherwise all of the work above is
    # thrown away without anything being displayed.
    fig.savefig(filename, bbox_inches='tight')
    print("Saved plot to: " + filename)

    # Release the figure, otherwise matplotlib keeps every figure alive
    # and the memory usage grows with each call.
    plt.close(fig)


def plot_layer_output(layer_output, image, x, session, filename='layer_output.png'):
    # Assume layer_output is a 4-dim tensor
    # e.g. output_conv1 or output_conv2.

    # Create a feed-dict which holds the single input image.
    # Note that TensorFlow needs a list of images,
    # so we just create a list with this one image.
    feed_dict = {x: [image]}

    # Retrieve the output of the layer after inputting this image.
    values = session.run(layer_output, feed_dict=feed_dict)

    # Get the lowest and highest values.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    values_min = np.min(values)
    values_max = np.max(values)

    # Number of image channels output by the conv. layer.
    num_images = values.shape[3]

    # Number of grid-cells to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = int(math.ceil(math.sqrt(num_images)))

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid image-channels.
        if i < num_images:
            # Get the images for the i'th output channel.
            img = values[0, :, :, i]

            # Plot image.
            ax.imshow(img, vmin=values_min, vmax=values_max,
                      interpolation='nearest', cmap='binary')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Write the figure to disk, otherwise all of the work above is
    # thrown away without anything being displayed.
    fig.savefig(filename, bbox_inches='tight')
    print("Saved plot to: " + filename)

    # Release the figure, otherwise matplotlib keeps every figure alive
    # and the memory usage grows with each call.
    plt.close(fig)


def plot_distorted_image(image, cls_true, distorted_images,
                         x, session, dataset):
    # Repeat the input image 9 times.
    image_duplicates = np.repeat(image[np.newaxis, :, :, :], 9, axis=0)

    # Create a feed-dict for TensorFlow.
    feed_dict = {x: image_duplicates}

    # Calculate only the pre-processing of the TensorFlow graph
    # which distorts the images in the feed-dict.
    result = session.run(distorted_images, feed_dict=feed_dict)

    # Plot the images.
    class_names = dataset['class_names']
    plot_images(images=result, cls_true=np.repeat(cls_true, 9),
                class_names=class_names, filename='distorted_image.png')


def plot_image(image, filename='image.png'):
    # Create figure with sub-plots.
    fig, axes = plt.subplots(1, 2)

    # References to the sub-plots.
    ax0 = axes.flat[0]
    ax1 = axes.flat[1]

    # Show raw and smoothened images in sub-plots.
    ax0.imshow(image, interpolation='nearest')
    ax1.imshow(image, interpolation='spline16')

    # Set labels.
    ax0.set_xlabel('Raw')
    ax1.set_xlabel('Smooth')

    # Write the figure to disk, otherwise all of the work above is
    # thrown away without anything being displayed.
    fig.savefig(filename, bbox_inches='tight')
    print("Saved plot to: " + filename)

    # Release the figure, otherwise matplotlib keeps every figure alive
    # and the memory usage grows with each call.
    plt.close(fig)
