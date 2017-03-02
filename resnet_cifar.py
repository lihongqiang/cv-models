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

(X_train, Y_train), (X_test, Y_test) = tflearn.datasets.cifar10.load_data(one_hot = True)

print X_train.shape, Y_train.shape

#%%

# Residual blocks
# 32 layers: n=5, 56 layers: n=9, 110 layers: n=18
n = 5

#%%
#实时图片前处理，每个channel去均值
preprocessing = tflearn.ImagePreprocessing()
preprocessing.add_featurewise_zero_center(per_channel=True)

#%%
#实时图片增加
augment = tflearn.ImageAugmentation()
augment.add_random_flip_leftright()
augment.add_random_crop(crop_shape=[32,32,3], padding=4)

#%%
with tf.Graph().as_default():
    
    input_data = tflearn.input_data(shape=[None, 32, 32, 3], 
                                    data_preprocessing=preprocessing,
                                    data_augmentation=augment)
    
    conv1 = tflearn.conv_2d(input_data, nb_filter=16, filter_size=3,
                            weight_decay=0.0001, regularizer='L2')
    
    resnet = tflearn.residual_bottleneck(conv1, nb_blocks=n, bottleneck_size=16,
                                         out_channels=16)
    
    resnet = tflearn.residual_bottleneck(resnet, nb_blocks=1, bottleneck_size=32,
                                         out_channels=32, downsample=True)
    resnet = tflearn.residual_bottleneck(resnet, nb_blocks=n-1, bottleneck_size=32,
                                         out_channels=32)
    
    
    resnet = tflearn.residual_bottleneck(resnet, nb_blocks=1, bottleneck_size=64,
                                         out_channels=64)
    resnet = tflearn.residual_bottleneck(resnet, nb_blocks=n-1, bottleneck_size=64)
    
    resnet = tflearn.batch_normalization(resnet)
    
    resnet = tflearn.activation(resnet, activation='relu')
    
    resnet = tflearn.global_avg_pool(resnet)
    
    fcn = tflearn.fully_connected(resnet, n_units=10, activation='softmax')
    
    momentum = tflearn.Momentum(learning_rate=0.1, lr_decay=0.1, decay_step=32000,
                                staircase=True)
    regression = tflearn.regression(fcn, optimizer=momentum,
                                    loss='categorical_crossentropy')
    model = tflearn.DNN(regression, tensorboard_dir='./tf-learn/resnet_cifar/logs', 
                        checkpoint_path='./tf-learn/resnet_cifar/checkpoints',
                        max_checkpoints=10, tensorboard_verbose=1)
    model.fit(X_train, Y_train, n_epoch=200, validation_set=(X_test, Y_test),
              show_metric=True, batch_size=128, shuffle=True, snapshot_step=500,
              snapshot_epoch=False, run_id='resnet-cifar')
    