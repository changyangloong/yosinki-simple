# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 15:27:44 2021

@author: holmes
"""

import numpy as np
from PIL import Image
import cv2
import random
import tensorflow as tf
import os
from six.moves import cPickle

os.environ["CUDA_VISIBLE_DEVICES"]= "1"

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.00001
SAVE_DIR = 'Model'
NUMPY_WEIGHT = 'save_weights.pkl'

def fc(
        x,
        num_in,
        num_out,
        name,
        relu=True,
        trainable=True
       ):
    """
    Fully connected layers 
    """
    with tf.variable_scope(name) as scope:
        w = tf.get_variable(
                    'w',
                    shape = [num_in,num_out],
                    trainable=trainable,
                    regularizer = tf.contrib.layers.l2_regularizer(scale=WEIGHT_DECAY)
                )
        b = tf.get_variable(
                    'b',
                    shape = [num_out],
                    trainable=trainable
                )
        
        matmul_result = tf.nn.xw_plus_b(x,w,b,name=scope.name)
        
    if relu:
        return tf.nn.relu(matmul_result)
    else:
        return matmul_result

# change input to variable, 1 sample = batch = 1
x = tf.get_variable(
    'x', shape = [1,3], trainable = True, 
    regularizer = tf.contrib.layers.l2_regularizer(scale=WEIGHT_DECAY),
    initializer = tf.random_uniform_initializer(0,1))

y = tf.placeholder(tf.int32,(1,))
y_onehot = tf.one_hot(y,2)


fc1 = fc( x = x,num_in = 3,num_out = 10,name = 'fc1')
logits = fc( x = fc1,num_in = 10,num_out = 2,name = 'output')

var_list = [v for v in tf.trainable_variables()]
regularizer = tf.contrib.layers.l2_regularizer(scale=WEIGHT_DECAY)
reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)

with tf.name_scope("loss"):
    softmax = tf.nn.softmax(logits)
    
    # not sure loss
    network_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=logits,labels=y_onehot)) 

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) 
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    gradient = optimizer.compute_gradients(network_loss,[x])
    
train_step = optimizer.apply_gradients(
    [(gc[0], gc[1]) for i, gc in enumerate(gradient)]) 


with open(NUMPY_WEIGHT,'rb') as fid:
    val = cPickle.load(fid)

custom_load_ops = []    
for var,v in zip(var_list[1:],val):
    custom_load_ops.append(tf.assign(var,v))

saver = tf.train.Saver(var_list=var_list, max_to_keep=0)   
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)


with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(tf.global_variables_initializer())
    
    sess.run(custom_load_ops)
    
    theloss,sm,g,var = sess.run(
                [network_loss,softmax,gradient,var_list],
                feed_dict={
                    y : np.array([0,])
                    }
            )
    print(theloss,sm,g,var[0])
    
    for i in range(1000):
        theloss,sm,g,_,var = sess.run(
                [network_loss,softmax,gradient,train_step,var_list],
                feed_dict={
                    y : np.array([0,])
                    }
            )
        sess.run(train_step,feed_dict={
                    y : np.array([0,])
                    })
        
        print(theloss,sm,g,var[0])
        print('\n')
     
    












