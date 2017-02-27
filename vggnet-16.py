#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 11:10:53 2017

@author: lhq
"""
#%%
import tflearn
import tensorflow as tf

X, Y = tflearn.datasets.oxflower17.load_data(resize_pics=(224, 224), shuffle=True, one_hot=True)

#%%

print X.shape, Y.shape

#%%

#%%
with tf.Graph().as_default():
    
    input_data = tflearn.input_data(shape=[None, 224, 224, 3])
    
    layer1 = tflearn.conv_2d(input_data, nb_filter=64, filter_size=3, activation='relu')
    layer1 = tflearn.conv_2d(layer1, nb_filter=64, filter_size=3, activation='relu')
    layer1 = tflearn.max_pool_2d(layer1, kernel_size=2, strides=2)
    
    layer2 = tflearn.conv_2d(layer1, nb_filter=128, filter_size=3, activation='relu')
    layer2 = tflearn.conv_2d(layer2, nb_filter=128, filter_size=3, activation='relu')
    layer2 = tflearn.max_pool_2d(layer2, kernel_size=2, strides=2)
    
    layer3 = tflearn.conv_2d(layer2, nb_filter=256, filter_size=3, activation='relu')
    layer3 = tflearn.conv_2d(layer3, nb_filter=256, filter_size=3, activation='relu')
    layer3 = tflearn.conv_2d(layer3, nb_filter=256, filter_size=3, activation='relu')
    layer3 = tflearn.max_pool_2d(layer3, kernel_size=2, strides=2)
    
    layer4 = tflearn.conv_2d(layer3, nb_filter=256, filter_size=3, activation='relu')
    layer4 = tflearn.conv_2d(layer4, nb_filter=256, filter_size=3, activation='relu')
    layer4 = tflearn.conv_2d(layer4, nb_filter=256, filter_size=3, activation='relu')
    layer4 = tflearn.max_pool_2d(layer4, kernel_size=2, strides=2)
    
    layer5 = tflearn.conv_2d(layer4, nb_filter=512, filter_size=3, activation='relu')
    layer5 = tflearn.conv_2d(layer5, nb_filter=512, filter_size=3, activation='relu')
    layer5 = tflearn.conv_2d(layer5, nb_filter=512, filter_size=3, activation='relu')
    layer5 = tflearn.max_pool_2d(layer5, kernel_size=2, strides=2)
    
    layer6 = tflearn.fully_connected(layer5, n_units=4096, activation='relu')
    layer6 = tflearn.dropout(layer6, 0.5)
    
    layer7 = tflearn.fully_connected(layer6, n_units=4096, activation='relu')
    layer7 = tflearn.dropout(layer7, 0.5)
    
    layer7 = tflearn.fully_connected(layer7, n_units=17, activation='softmax')
    
    regression = tflearn.regression(layer7, loss='categorical_crossentropy',
                                    optimizer='rmsprop', learning_rate=0.001)
    
    model = tflearn.DNN(regression, tensorboard_dir='/Users/lhq/Workspace/python/tf-learn/vggnet-16/logs',
                    tensorboard_verbose=2, checkpoint_path='/Users/lhq/Workspace/python/tf-learn/vggnet-16/checkpoints',
                    max_checkpoints=3)
    model.load
    model.fit(X, Y, n_epoch=1, show_metric=True, batch_size=32, shuffle=True, snapshot_step=500,
              snapshot_epoch=False, run_id='vgg16')
    model.save('/Users/lhq/Workspace/python/tf-learn/vggnet-16/vgg16')
    