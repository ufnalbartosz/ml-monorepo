import numpy as np
import pickle
import os
import sys
import urllib
import tarfile
import zipfile


from project_cnn.dataset import one_hot_encoded
data_path = "data/CIFAR-100/"
data_url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"

# Labels names.
food_containers = ['bottle', 'bowl', 'can', 'cup', 'plate']
fruit_and_vegetables = ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper']
household_electrical_devices = ['clock', 'keyboard',
                                'lamp', 'telephone', 'television']
household_furniture = ['bed', 'chair', 'couch', 'table', 'wardrobe']

labels = (food_containers + fruit_and_vegetables +
          household_electrical_devices + household_furniture)

labels = sorted(labels)

# Width and height of each image.
img_size = 32

# Number of channels in each image, 3 channels: Red, Green, Blue.
num_channels = 3

# Length of an image when flattened to a 1-dim array.
img_size_flat = img_size * img_size * num_channels

# Number of classes.
num_classes = len(labels)


def _unpickle(filename):
    # Create full path for the file.
    file_path = os.path.join(data_path, 'cifar-100-python', filename)

    print("Loading data: " + file_path)

    with open(file_path, mode='rb') as fp:
        data = pickle.load(fp)

    return data


def _convert_images(raw):
    """
    Convert images from the CIFAR-10 format and
    return a 4-dim array with shape: [image_number, height, width, channel]
    where the pixels are floats between 0.0 and 1.0.
    """

    # Convert the raw images from the data-files to floating-points.
    raw_float = np.array(raw, dtype=float) / 255.0

    # Reshape the array to 4-dimensions.
    images = raw_float.reshape([-1, num_channels, img_size, img_size])

    # Reorder the indices of the array.
    images = images.transpose([0, 2, 3, 1])

    return images


def _convert_labels(fine_labels):
    mask, class_names = create_mask_class_names_to_skip(fine_labels)

    converted_labels = []
    fine_labels = fine_labels[mask, ...]
    for id in fine_labels:
        label_name = class_names[id]
        new_id = labels.index(label_name)
        converted_labels.append(new_id)

    converted_labels = np.array(converted_labels)

    return converted_labels, mask


def _load_data(filename):
    """
    Load a pickled data-file from the CIFAR-10 data-set
    and return the converted images (see above) and the class-number
    for each image.
    """

    # Load the pickled data-file.
    data = _unpickle(filename)

    # Get the raw images.
    raw_images = data['data']

    # Get the class-numbers for each image. Convert to numpy-array.
    fine_labels = np.array(data['fine_labels'])

    coverted_labels, mask = _convert_labels(fine_labels)

    # Convert the images.
    images = _convert_images(raw_images)
    images = images[mask, ...]

    return images, coverted_labels


########################################################################
# Public functions that you may call to download the data-set from
# the internet and load the data into memory.


def _print_download_progress(count, block_size, total_size):
    # Percentage completion.
    pct_complete = float(count * block_size) / total_size

    # Status-message. Note the \r which means the line should overwrite itself.
    msg = "\r- Download progress: {0:.1%}".format(pct_complete)

    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()


def _maybe_download_and_extract(url, download_dir):
    # Filename for saving the file downloaded from the internet.
    # Use the filename from the URL and add it to the download_dir.
    filename = url.split('/')[-1]
    file_path = os.path.join(download_dir, filename)

    # Check if the file already exists.
    # If it exists then we assume it has also been extracted,
    # otherwise we need to download and extract it now.
    if not os.path.exists(file_path):
        # Check if the download directory exists, otherwise create it.
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        # Download the file from the internet.
        file_path, _ = urllib.urlretrieve(url=url,
                                          filename=file_path,
                                          reporthook=_print_download_progress)

        print()
        print("Download finished. Extracting files.")

        if file_path.endswith(".zip"):
            # Unpack the zip-file.
            zipfile.ZipFile(file=file_path, mode="r").extractall(download_dir)
        elif file_path.endswith((".tar.gz", ".tgz")):
            # Unpack the tar-ball.
            tarfile.open(name=file_path, mode="r:gz").extractall(download_dir)

        print("Done.")
    else:
        print("Data has apparently already been downloaded and unpacked.")


def maybe_download_and_extract():
    _maybe_download_and_extract(url=data_url, download_dir=data_path)


def load_data(filename):
    images, cls = _load_data(filename=filename)

    return images, cls, one_hot_encoded(
        class_numbers=cls, num_classes=num_classes)


def load_class_names():
    # The meta-file holds two lists: 'fine_label_names' (100 classes) and
    # 'coarse_label_names' (20 super-classes). Ask for the one we need
    # instead of relying on the ordering of the dict-keys.
    meta = _unpickle(filename="meta")
    return meta['fine_label_names']


def create_mask_class_names_to_skip(fine_labels):
    class_names = load_class_names()

    mask = []
    for label in fine_labels:
        if class_names[label] in labels:
            mask.append(True)
        else:
            mask.append(False)

    return mask, class_names
