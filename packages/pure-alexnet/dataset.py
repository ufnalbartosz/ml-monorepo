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
    indexes_filepath = '17flowers/indexes.pickle'

    dir_list = ['saves', 'logs', 'checkpoints']
    for _dir in dir_list:
        if not os.path.isdir(_dir):
            os.mkdir(_dir)

    if os.path.exists(dataset_filepath):
        print("Loading pickle dataset")
        with open(dataset_filepath, 'rb') as fp:
            dataset_dict = pickle.load(fp)
    else:
        print("Creating pickle dataset")
        import tflearn.datasets.oxflower17 as oxflower17

        raw_images, raw_labels = oxflower17.load_data(one_hot=True, resize_pics=(227, 227))

        if os.path.exists(indexes_filepath):
            with open(indexes_filepath, 'rb') as fp:
                indexes_dict = pickle.load(fp)
        else:
            from indexes import create_pickle_indexes
            indexes_dict = create_pickle_indexes()
            with open(indexes_filepath, 'wb') as fp:
                pickle.dump(indexes_dict, fp, pickle.HIGHEST_PROTOCOL)

        test_indexes = indexes_dict['test_indexes']
        train_indexes = indexes_dict['train_indexes']

        print(len(test_indexes), 'test_indexes')
        print(test_indexes)
        test_images = raw_images[test_indexes]
        test_labels = raw_labels[test_indexes]

        print(len(test_images), 'test images')

        print(len(train_indexes), 'train_indexes')
        train_images = raw_images[train_indexes]
        train_labels = raw_labels[train_indexes]

        dataset_dict = {}
        dataset_dict.setdefault('test_images', test_images)
        print(test_images.shape)
        dataset_dict.setdefault('test_labels', test_labels)
        print(test_labels.shape)
        dataset_dict.setdefault('train_images', train_images)
        print(train_images.shape)
        dataset_dict.setdefault('train_labels', train_labels)
        print(train_labels.shape)

        with open(dataset_filepath, 'wb') as fp:
            pickle.dump(dataset_dict, fp, pickle.HIGHEST_PROTOCOL)

    return dataset_dict


if __name__ == '__main__':
    maybe_download_and_extract()
