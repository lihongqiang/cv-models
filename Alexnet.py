#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 16:15:28 2017

@author: lhq
"""
#%%
import tflearn

(X, Y) = tflearn.datasets.oxflower17.load_data(resize_pics=(227,227), one_hot=True)

#%%
print X.shape

#%%

import tensorflow as tf

with tf.Graph().as_default():
    
    input_data = tflearn.input_data(shape=[None, 227,227,3])
    conv1 = tflearn.conv_2d(input_data, nb_filter=96, filter_size=11, strides=4,
                            activation='relu')
    maxpool1 = tflearn.max_pool_2d(conv1, kernel_size=3, strides=2)
    norm1 = tflearn.local_response_normalization(maxpool1)
    
    
    conv2 = tflearn.conv_2d(norm1, nb_filter=256, filter_size=5, strides=1, 
                            activation='relu')
    maxpool2 = tflearn.max_pool_2d(conv2, kernel_size=3,
                                   strides=2)
    norm2 = tflearn.local_response_normalization(maxpool2)
    
    
    conv3 = tflearn.conv_2d(norm2, nb_filter=384, filter_size=3, strides=1,
                            activation='relu')
    
    conv4 = tflearn.conv_2d(conv3, nb_filter=384, filter_size=3, strides=1,
                            activation='relu')
    
    conv5 = tflearn.conv_2d(conv4, nb_filter=256, filter_size=3, strides=1,
                            activation='relu')
    maxpool3 = tflearn.max_pool_2d(conv5, kernel_size=3, strides=2)
    norm3 = tflearn.local_response_normalization(maxpool3)
    
    fucn6 = tflearn.fully_connected(norm3, n_units=2048, activation='tanh')
    drop1 = tflearn.dropout(fucn6, 0.5)
    
    fucn7 = tflearn.fully_connected(fucn6, n_units=2048, activation='tanh')
    drop2 = tflearn.dropout(fucn7, 0.5)
    
    fucn8 = tflearn.fully_connected(fucn7, n_units=17, activation='softmax')
    
    regression = tflearn.regression(fucn8, loss='categorical_crossentropy',
                                    optimizer='adam', learning_rate=0.001)
    
    model = tflearn.DNN(regression, tensorboard_dir='./tf-learn/Alexnet/logs',
                        checkpoint_path='./tf-learn/Alexnet/checkpoint/', 
                        tensorboard_verbose=2, max_checkpoints=3)
    
    model.fit(X, Y, n_epoch=1, validation_set=0.1, show_metric=True,
              snapshot_epoch=False, snapshot_step=200, batch_size=64,
              run_id='alexnet_oxflowers17')
    model.save('./tf-learn/Alexnet/model_epoch_10')
    