# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

from dataset import DataSet
data = DataSet().maybe_download_and_extract()

# Building 'AlexNet'
network = input_data(shape=[None, 227, 227, 3], name='input')
network = conv_2d(network, 96, 7, strides=1,
                  activation='relu', name='conv1_7_7')
network = max_pool_2d(network, 3, strides=2, name='max_pool1_3_3_2')
# network = local_response_normalization(network)
network = conv_2d(network, 128, 3, activation='relu', name='conv2_3_3')
# network = conv_2d(network, 128, 3, activation='relu', name='conv3_3_3')
network = max_pool_2d(network, 3, strides=2, name='max_pool2_3_3_2')
# network = local_response_normalization(network)
network = conv_2d(network, 128, 3, activation='relu', name='conv3_3_3')
network = conv_2d(network, 128, 3, activation='relu', name='conv4_3_3')
# network = conv_2d(network, 256, 3, activation='relu', name='conv5_3_3')
network = max_pool_2d(network, 3, strides=2, name='max_pool3_3_3_2')
# network = local_response_normalization(network)
network = fully_connected(
    network, 2048, activation='relu', name='fully_connected_1_relu')
network = dropout(network, 0.5, name='dropout_1_05')
network = fully_connected(
    network, 2048, activation='relu', name='fully_connected_2_relu')
network = dropout(network, 0.5, name='dropout_2_05')
network = fully_connected(
    network, 17, activation='softmax', name='fully_connected_3_softmax')
network = regression(network, optimizer='momentum',
                     loss='categorical_crossentropy',
                     learning_rate=0.001,
                     name='target')
model_name = '3_3'
# Training
model = tflearn.DNN(network,
                    checkpoint_path='checkpoints/{}'.format(model_name),
                    tensorboard_verbose=3,
                    tensorboard_dir='logs')

# model.load('saves/{}'.format(model_name))

model.fit({'input': data['train_images']},
          {'target': data['train_labels']},
          n_epoch=150,
          validation_set=({'input': data['valid_images']}, {
                          'target': data['valid_labels']}),
          shuffle=True,
          show_metric=True,
          batch_size=64,
          snapshot_step=500,
          snapshot_epoch=False,
          run_id='model')

predicted_labels = model.predict(data['test_images'])

pred_labels = []
for row in predicted_labels:
    pred_labels.append(np.argmax(row))
true_labels = []
for row in data['test_labels']:
    true_labels.append(np.argmax(row))
accuracy = np.sum(np.equal(pred_labels, true_labels))

num_test_images = data['test_images'].shape[0]
print(accuracy / len(true_labels))
print('Saving model...')
model.save('saves/{}'.format(model_name))
