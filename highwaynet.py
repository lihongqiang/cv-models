#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 18:16:27 2017

@author: lhq
"""
#%%
import tflearn
import tensorflow as tf

X, Y, X_test, Y_test = tflearn.datasets.mnist.load_data(one_hot=True)

#%%

with tf.Graph().as_default():
    
    input_data = tflearn.input_data(shape=[None, 28*28])
    layer1 = tflearn.fully_connected(input_data, 64, activation='elu',
                                     regularizer='L2', weight_decay=0.001)
    
    dense = layer1
    for i in range(10):
        dense = tflearn.highway(dense, 64, activation='elu',
                                regularizer='L2', weight_decay=0.001,
                                transform_dropout=0.8)
    
    softmax = tflearn.fully_connected(dense, 10, activation='softmax')
    
    sgd = tflearn.optimizers.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)
    topk = tflearn.metrics.top_k(3)
    regression = tflearn.regression(softmax, loss='categorical_crossentropy',
                                    optimizer=sgd, metric=topk)
    
    model = tflearn.DNN(regression, tensorboard_verbose=0)
    model.fit(X, Y, n_epoch=20, validation_set=(X_test, Y_test), 
              show_metric=True, run_id='high_dense_model')