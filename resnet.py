#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 14:29:04 2017

@author: lhq
"""

#%%
import tflearn
import tensorflow

#%%

X_train, Y_train, X_test, Y_test = tflearn.datasets.mnist.load_data( one_hot = True)

print X_train.shape, Y_train.shape

#%%
X_train.reshape([-1, 28, 28, 1])
X_test.reshape([-1, 28, 28, 1])

X_train, mean = tflearn.data_utils.featurewise_zero_center(X_train)


#%%
with tf.Graph().as_default():
    
    input_data = tflearn.input_data(shape=[None, 28, 28, 1])
    
    conv1 = tflearn.conv_2d(input_data, nb_filter=64, filter_size=3,
                            bias=False, activation='relu')
    
    resnet = tflearn.residual_bottleneck(conv1, nb_blocks=3, bottleneck_size=16,
                                         out_channels=64)
    
    resnet = tflearn.residual_bottleneck(resnet, nb_blocks=3, bottleneck_size=32,
                                         out_channels=128, downsample=True)
    
    resnet = tflearn.residual_bottleneck(resnet, nb_blocks=3, bottleneck_size=32,
                                         out_channels=128)
    
    resnet = tflearn.residual_bottleneck(resnet, nb_blocks=3, bottleneck_size=64,
                                         out_channels=256, downsample=True)
    
    resnet = tflearn.batch_normalization(resnet)
    
    resnet = tflearn.activation(resnet, activation='relu')
    
    resnet = tflearn.global_avg_pool(resnet)
    
    fcn = tflearn.fully_connected(resnet, n_units=10, activation='softmax')
    
    regression = tflearn.regression(fcn, optimizer='adam', batch_size=64,
                                    learning_rate=0.1, loss='categorical_crossentropy')
    model = tflearn.DNN(regression, tensorboard_dir='./tf-learn/resnet_mnist/logs', 
                        checkpoint_path='./tf-learn/resnet_mnist/checkpoints',
                        max_checkpoints=10, tensorboard_verbose=1)
    model.fit(X_train, Y_train, n_epoch=1, validation_set=(X_test, Y_test),
              show_metric=True, batch_size=256, run_id='resnet-mnist')
    