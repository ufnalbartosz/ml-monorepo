# -*- coding: utf-8 -*-

""" AlexNet.
Applying 'AlexNet' to the 20-class subset of CIFAR-100 selected in loader.py.
References:
    - Alex Krizhevsky, Ilya Sutskever & Geoffrey E. Hinton. ImageNet
    Classification with Deep Convolutional Neural Networks. NIPS, 2012.
    - CIFAR-100 Dataset. Alex Krizhevsky.
Links:
    - [AlexNet Paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
    - [CIFAR-100 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
"""

from __future__ import division, print_function, absolute_import
import os

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

from loader import img_size, num_channels, num_classes
from prepare_dataset import maybe_download_and_extract

dataset = maybe_download_and_extract()
X = dataset['train_images']
Y = dataset['train_labels']

# Use the validation-set for monitoring during training. The test-set must
# stay untouched until the final measurement, otherwise picking a snapshot
# based on it leaks the test-set into model selection.
X_valid = dataset['valid_images']
Y_valid = dataset['valid_labels']

# Building 'AlexNet'
network = input_data(shape=[None, img_size, img_size, num_channels])
network = conv_2d(network, 96, 3, strides=1, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 256, 5, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, num_classes, activation='softmax')
network = regression(network, optimizer='momentum',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# TensorFlow does not create the directories it writes checkpoints to, so
# make them here. Keep them under 'logs/' like the inception model does -
# that path is already covered by .gitignore.
logdir = 'logs'
checkpoint_dir = os.path.join(logdir, 'alexnet_checkpoints')
if not os.path.exists(logdir):
    os.mkdir(logdir)
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)
checkpoint_path = os.path.join(checkpoint_dir, 'model_alexnet')

# Training
model = tflearn.DNN(network,
                    checkpoint_path=checkpoint_path,
                    max_checkpoints=1,
                    tensorboard_verbose=2,
                    tensorboard_dir=logdir)

model.fit(X, Y,
          validation_set=(X_valid, Y_valid),
          batch_size=64,
          n_epoch=100,
          shuffle=True,
          show_metric=True,
          snapshot_step=500,
          snapshot_epoch=False,
          run_id='model_alexnet')

model.save(os.path.join(logdir, 'alexnet_model_save'))
