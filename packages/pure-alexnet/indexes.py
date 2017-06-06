import numpy as np


def create_pickle_indexes():
    test_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                    30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                    44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                    58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
                    72, 73, 74, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86,
                    87, 88, 89, 91, 92, 93, 94, 95, 96, 97, 99, 100, 102,
                    103, 104, 105, 106, 109, 110, 111, 112, 113, 114, 115,
                    116, 119, 120, 121, 122, 124, 126, 131, 132, 137, 138,
                    139, 140, 144, 149, 158, 166, 169, 173, 175, 178, 187,
                    193, 196, 214, 215]

    images_number = 1360
    train_indexes = [idx for idx in range(images_number) if idx not in test_indexes]

    indexes_dict = {
        'test_indexes': test_indexes,
        'train_indexes': train_indexes
    }
    return indexes_dict


def genereate_test_indexes(raw_images, raw_labels):
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

    return test_indexes
