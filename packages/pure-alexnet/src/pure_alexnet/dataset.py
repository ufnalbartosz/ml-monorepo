import pickle
import os
import shutil

import numpy as np

class DataSet():

    def __init__(self):
        self.dataset_filepath = '17flowers/dataset.pickle'
        self.labels = [
            'Buttercup', 'ColtsFoot', 'Daffodil', 'Daisy', 'Dandelion',
            'Firitillary', 'Iris', 'Pansy', 'Sunflower', 'Windflower',
            'Snowdrop', 'LilyValley', 'Bluebell', 'Crocus', 'Tigerlily',
            'Cowslip'
        ]

    def train_test_valid_split(self, raw_labels):
        images_number = raw_labels.shape[0]
        class_number = raw_labels.shape[1]  # 17

        test_label_counter = {}
        for class_ in range(class_number):
            test_label_counter[class_] = int(
                images_number / class_number * 0.2)  # 16

        valid_label_counter = {}
        for class_ in range(class_number):
            valid_label_counter[class_] = int(
                images_number / class_number * 0.1)  # 8

        test_indexes = []
        valid_indexes = []
        train_indexes = []

        for it, label in enumerate(raw_labels):
            index = np.argmax(label)

            if test_label_counter[index] > 0:
                test_label_counter[index] -= 1
                test_indexes.append(it)
            elif valid_label_counter[index] > 0:
                valid_label_counter[index] -= 1
                valid_indexes.append(it)
            else:
                train_indexes.append(it)

        return train_indexes, test_indexes, valid_indexes

    def maybe_download_and_extract(self):
        dir_list = ['saves', 'logs', 'checkpoints']
        for _dir in dir_list:
            if not os.path.isdir(_dir):
                print("Creating {} directory...".format(_dir))
                os.mkdir(_dir)

        if os.path.exists(self.dataset_filepath):
            print("Loading pickle dataset")
            with open(self.dataset_filepath, 'rb') as fp:
                dataset_dict = pickle.load(fp)
        else:
            print("Creating pickle dataset")
            import tflearn.datasets.oxflower17 as oxflower17

            raw_images, raw_labels = oxflower17.load_data(
                one_hot=True, resize_pics=(227, 227))

            train_ids, test_ids, valid_ids = self.train_test_valid_split(
                raw_labels)

            dataset_dict = {}
            dataset_dict.setdefault('train_images', raw_images[train_ids])
            dataset_dict.setdefault('train_labels', raw_labels[train_ids])
            dataset_dict.setdefault('test_images', raw_images[test_ids])
            dataset_dict.setdefault('test_labels', raw_labels[test_ids])
            dataset_dict.setdefault('valid_images', raw_images[valid_ids])
            dataset_dict.setdefault('valid_labels', raw_labels[valid_ids])

            with open(self.dataset_filepath, 'wb') as fp:
                pickle.dump(dataset_dict, fp, pickle.HIGHEST_PROTOCOL)

            self.cleanup()

        return dataset_dict

    def cleanup(self):
        dataset_dirname = os.path.dirname(self.dataset_filepath)

        jpg_dir = os.path.join(dataset_dirname, 'jpg')
        if os.path.exists(jpg_dir) and os.path.isdir(jpg_dir):
            shutil.rmtree(jpg_dir)

        for extension in ['.tgz', '.pkl']:
            filename = os.path.join(dataset_dirname, '17flowers' + extension)
            if os.path.exists(filename):
                os.remove(filename)
