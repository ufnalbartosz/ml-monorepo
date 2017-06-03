# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

from dataset import maybe_download_and_extract
dataset = maybe_download_and_extract()
train_images = dataset['train_images']
train_labels = dataset['train_labels']
test_images = dataset['test_images']
test_labels = dataset['test_labels']

# Building 'AlexNet'
network = input_data(shape=[None, 227, 227, 3])
network = conv_2d(network, 96, 11, strides=4, activation='relu')
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
network = fully_connected(network, 17, activation='softmax')
network = regression(network, optimizer='momentum',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Training
model = tflearn.DNN(network,
                    checkpoint_path='checkpoints/model',
                    tensorboard_verbose=2,
                    tensorboard_dir='logs')
# model.load('saves/model')
model.load('checkpoints/model-4000')

# model.fit(X, Y,
#           n_epoch=1,
#           validation_set=0.1,
#           shuffle=True,
#           show_metric=True,
#           batch_size=64,
#           snapshot_step=500,
#           snapshot_epoch=False,
#           run_id='model')

tx = test_images[0]
print(np.argmax(test_labels[0]))
plt.imshow(tx)
plt.show()
print(tx.shape)
predict_labels = model.predict([tx])

print(predict_labels[0])
print(np.argmax(predict_labels[0]))
model.save('saves/model')
