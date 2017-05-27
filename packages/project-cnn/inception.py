from __future__ import division, print_function, absolute_import
import os

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

# Data loading and preprocessing
from loader import num_classes
from prepare_dataset import maybe_download_and_extract

dataset = maybe_download_and_extract()
X = dataset['train_images']
Y = dataset['train_labels']

# Use the validation-set for monitoring during training. The test-set must
# stay untouched until the final measurement, otherwise picking a snapshot
# based on it leaks the test-set into model selection.
X_valid = dataset['valid_images']
Y_valid = dataset['valid_labels']

# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)

# Convolutional network building
network = input_data(shape=[None, 32, 32, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug,
                     name='input')

# 1st layer
conv1_3_3 = conv_2d(network, 32, 3, activation='relu', name='conv1_3_3')
pool1_3_3 = max_pool_2d(conv1_3_3, 2)
pool1_3_3 = local_response_normalization(pool1_3_3)

# 2nd layer
# incpetion2a
inception2a_1_1 = conv_2d(pool1_3_3, 64, 1, activation='relu', name='inception2a_1_1')

# inpcetion2b
inception2b_3_3_reduce = conv_2d(pool1_3_3, 64, 1, activation='relu', name='inception2b_3_3_reduce')
inception2b_3_3 = conv_2d(inception2b_3_3_reduce, 32, 3, activation='relu', name='inception2b_3_3')

# inception2c
inception2c_5_5_reduce = conv_2d(pool1_3_3, 64, 1, activation='relu', name='inception2c_5_5_reduce')
inception2c_5_5 = conv_2d(inception2c_5_5_reduce, 32, 5, activation='relu', name='inception2c_5_5')

# inception2d
inception2d_pool = max_pool_2d(pool1_3_3, kernel_size=3, strides=1, name='inception2d_pool')
inception2d_pool_1_1 = conv_2d(inception2d_pool, 32, 1, activation='relu', name='inception2d_pool_1_1')

# inception2_output
inception2_output = merge([inception2a_1_1, inception2b_3_3, inception2c_5_5, inception2d_pool_1_1],
                          mode='concat', axis=3)

# 3rd layer
# incpetion3a
inception3a_1_1 = conv_2d(inception2_output, 64, 1, activation='relu', name='inception3a_1_1')

# inpcetion3b
inception3b_3_3_reduce = conv_2d(inception2_output, 64, 1, activation='relu', name='inception3b_3_3_reduce')
inception3b_3_3 = conv_2d(inception3b_3_3_reduce, 32, 3, activation='relu', name='inception3b_3_3')

# inception3c
inception3c_5_5_reduce = conv_2d(inception2_output, 64, 1, activation='relu', name='inception3c_5_5_reduce')
inception3c_5_5 = conv_2d(inception3c_5_5_reduce, 32, 5, activation='relu', name='inception3c_5_5')

# inception3d
inception3d_pool = max_pool_2d(inception2_output, kernel_size=3, strides=1, name='inception3d_pool')
inception3d_pool_1_1 = conv_2d(inception3d_pool, 32, 1, activation='relu', name='inception3d_pool_1_1')

# inception3_output
inception3_output = merge([inception3a_1_1, inception3b_3_3, inception3c_5_5, inception3d_pool_1_1],
                          mode='concat', axis=3)


# 4th layer
pool4_7_7 = avg_pool_2d(inception3_output, kernel_size=7, strides=1)
pool4_7_7 = dropout(pool4_7_7, 0.5)
loss = fully_connected(pool4_7_7, num_classes, activation='softmax')

# sgd = tflearn.optimizers.SGD(learning_rate=0.001, lr_decay=0.96, decay_step=100)

network = regression(loss, optimizer='momentum',
                     loss='categorical_crossentropy',
                     learning_rate=0.001,
                     name='target')

logdir = 'logs'
checkpoint_dir = os.path.join(logdir, 'inception_checkpoints')
if not os.path.exists(logdir):
    os.mkdir(logdir)
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)
checkpoint_path = os.path.join(checkpoint_dir, "checkpoint")

# Train using classifier
model = tflearn.DNN(network,
                    tensorboard_verbose=2,
                    checkpoint_path=checkpoint_path,
                    tensorboard_dir=logdir)

model.fit({'input': X}, {'target': Y},
          validation_set=({'input': X_valid}, {'target': Y_valid}),
          n_epoch=1000,
          shuffle=True,
          show_metric=True,
          batch_size=96,
          snapshot_step=500,
          snapshot_epoch=False,
          run_id='inception_model')

model.save('logs/inception_model_save')
