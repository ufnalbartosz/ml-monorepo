import numpy as np
import pickle
import os


def create_readable_labels():
    flower_dict = {}
    flower_dict.setdefault(1, 'Buttercup')
    flower_dict.setdefault(2, 'ColtsFoot')
    flower_dict.setdefault(3, 'Daffodil')
    flower_dict.setdefault(4, 'Daisy')
    flower_dict.setdefault(5, 'Dandelion')
    flower_dict.setdefault(6, 'Firitillary')
    flower_dict.setdefault(7, 'Iris')
    flower_dict.setdefault(8, 'Pansy')
    flower_dict.setdefault(9, 'Sunflower')
    flower_dict.setdefault(10, 'Windflower')
    flower_dict.setdefault(11, 'Snowdrop')
    flower_dict.setdefault(12, 'LilyValley')
    flower_dict.setdefault(13, 'Bluebell')
    flower_dict.setdefault(14, 'Crocus')
    flower_dict.setdefault(15, 'Tigerlily')
    flower_dict.setdefault(16, 'Tulip')
    flower_dict.setdefault(17, 'Cowslip')
    return flower_dict


def maybe_download_and_extract():
    dataset_filepath = '17flowers/dataset.pickle'

    if os.path.exists(dataset_filepath):
        with open(dataset_filepath, 'rb') as fp:
            dataset_dict = pickle.load(fp)
    else:
        import tflearn.datasets.oxflower17 as oxflower17

        raw_images, raw_labels = oxflower17.load_data(one_hot=True, resize_pics=(227, 227))

        images_number = raw_images.shape[0]
        class_number = raw_labels.shape[1]  # 17

        label_counter = {}
        for number in range(class_number):
            label_counter[number] = images_number / class_number * 0.1  # 8

        test_indexes = []
        for it, (image, label) in enumerate(zip(raw_images, raw_labels)):
            index = np.argmax(label)

            if label_counter[index] > 0:
                label_counter[index] -= 1
                test_indexes.append(it)

        test_images = raw_images[test_indexes]
        test_labels = raw_labels[test_indexes]

        train_indexes = [idx for idx in range(images_number) if idx not in test_images]
        train_images = raw_images[train_indexes]
        train_labels = raw_images[train_indexes]

        dataset_dict = {}
        dataset_dict.setdefault('test_images', test_images)
        dataset_dict.setdefault('test_labels', test_labels)
        dataset_dict.setdefault('train_images', train_images)
        dataset_dict.setdefault('train_labels', train_labels)

        print len(test_indexes)
        print len(train_indexes)

        with open(dataset_filepath, 'wb') as fp:
            pickle.dump(dataset_dict, fp, pickle.HIGHEST_PROTOCOL)

    return dataset_dict


if __name__ == '__main__':
    maybe_download_and_extract()
