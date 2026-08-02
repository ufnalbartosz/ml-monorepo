import pickle
import numpy as np
import os

# local imports
from project_cnn import loader


def maybe_download_and_extract():
    try:
        dataset = load_pickle_dataset('data/dataset.pickle')
    except IOError:
        dataset = create_pickle_dataset()

    return dataset


def load_pickle_dataset(filename):
    filename_fullpath = os.path.join(os.getcwd(), filename)

    if os.path.exists(filename_fullpath):
        print('Data has been already downloaded and unpacked.')
        with open(filename_fullpath, 'rb') as fp:
            return pickle.load(fp)
    else:
        raise IOError("File '{}' does not exists.".format(filename))


def create_pickle_dataset():
    loader.maybe_download_and_extract()

    test_images, test_cls, test_labels = loader.load_data("test")
    dataset = split_test_dataset(test_images, test_cls, test_labels)

    train_images, train_cls, train_labels = loader.load_data("train")
    dataset.setdefault('train_images', train_images)
    dataset.setdefault('train_labels', train_labels)
    dataset.setdefault('train_cls', train_cls)

    dataset.setdefault('class_names', loader.labels)

    to_pickle(dataset)
    return dataset


def split_test_dataset(test_images, test_cls, test_labels):
    halved = np.array([], dtype=int)

    for i in range(len(test_labels[0])):
        halved = np.concatenate((halved, np.where(test_cls == i)[0][::2]))
    mask = np.sort(halved)

    valid_images = test_images[mask, ...]
    valid_labels = test_labels[mask, ...]
    valid_cls = test_cls[mask, ...]

    # Everything that did not go into the validation-set stays in the
    # test-set. Derive it from the actual size of the test-set instead of
    # a hard-coded 2000, which silently breaks when the list of labels
    # in loader.py changes.
    reversed_mask = np.setdiff1d(np.arange(len(test_cls)), mask)

    test_images = test_images[reversed_mask, ...]
    test_labels = test_labels[reversed_mask, ...]
    test_cls = test_cls[reversed_mask, ...]

    dataset_dict = {
        'test_images': test_images,
        'test_labels': test_labels,
        'test_cls': test_cls,
        'valid_images': valid_images,
        'valid_labels': valid_labels,
        'valid_cls': valid_cls,
    }

    return dataset_dict


def to_pickle(dataset_dict):

    pickle_path = 'data/dataset.pickle'
    pickle_fullpath = os.path.join(os.getcwd(), pickle_path)

    # Save file with dataset pickle format.
    if not os.path.exists(pickle_fullpath):
        with open(pickle_fullpath, 'wb') as fp:
            pickle.dump(dataset_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        raise IOError("File '{}' already exists.".format(pickle_fullpath))


if __name__ == '__main__':
    maybe_download_and_extract()
