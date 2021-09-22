# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 15:26:04 2021

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

LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.0000001
SAVE_DIR = 'Model'
MODEL_NAME = 'model_final_new_2.ckpt'
DATA_NUM = 10000
TEST_NUM = 1000
BATCH = 50
EPOCH = 30
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

# setting up training data, red and blue color pallet in HSV, 0-red 1-blue
# 10000 sample, with 9000 training, 1000 test
# input size per sample is (1,3)
color_data_red = np.concatenate([np.random.randint(161,179,size=(DATA_NUM//2,1)),
                np.random.randint(155,255,size=(DATA_NUM//2,1)),
                np.random.randint(84,255,size=(DATA_NUM//2,1))],axis=1)

color_data_blue = np.concatenate([np.random.randint(100,140,size=(DATA_NUM//2,1)),
                np.random.randint(150,255,size=(DATA_NUM//2,1)),
                np.random.randint(0,255,size=(DATA_NUM//2,1))],axis=1)

# # just for check if the image correct
# i = 0
# img = np.ones((300,300,3),dtype=np.uint8)
# img[:,:,0] = img[:,:,0] * color_data_blue[i,0]
# img[:,:,1] = img[:,:,1] * color_data_blue[i,1]
# img[:,:,2] = img[:,:,2] * color_data_blue[i,2]
# img = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)

color_data = np.concatenate([color_data_red/255,color_data_blue/255])

labels = np.concatenate([np.zeros((DATA_NUM//2,),dtype=np.int32),
                        np.ones((DATA_NUM//2,),dtype=np.int32)])
idx = np.arange(DATA_NUM)
random.shuffle(idx)
train_idx = idx[0:(DATA_NUM-TEST_NUM)]
test_idx = idx[(DATA_NUM-TEST_NUM):]


# network implementation
batch = BATCH

x = tf.placeholder(tf.float32,(batch,3))
y = tf.placeholder(tf.int32,(batch,))
y_onehot = tf.one_hot(y,2) # turn label to 1,0 notation

is_training = tf.placeholder(tf.bool)

fc1 = fc( x = x,num_in = 3,num_out = 10,name = 'fc1')
# dropout1 = tf.nn.dropout(fc1,0.5)

logits = fc( x = fc1,num_in = 10,num_out = 2,name = 'output')

var_list = [v for v in tf.trainable_variables()]
regularizer = tf.contrib.layers.l2_regularizer(scale=WEIGHT_DECAY)
reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)

with tf.name_scope("loss"):
    network_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=logits,labels=y_onehot)) 
    
    total_loss = network_loss + reg_term

    softmax = tf.nn.softmax(logits)
    prediction = tf.argmax(softmax,axis=-1, output_type=tf.int32)
    match = tf.equal(prediction,tf.argmax(y_onehot,1, output_type=tf.int32))
    accuracy = tf.reduce_mean(tf.cast(match,tf.float32))

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) 
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    gradient = optimizer.compute_gradients(total_loss,var_list)
    
# train_step = optimizer.apply_gradients(gradient)
train_step = optimizer.apply_gradients(
                    [(gc[0], gc[1]) for i, gc in enumerate(gradient)]) 


saver = tf.train.Saver(var_list=var_list, max_to_keep=0)    
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess, 'Model/model_final.ckpt')  
    
    for j in range(EPOCH):
        for i in range((DATA_NUM-TEST_NUM)//batch):
            loss,acc,sm,_ = sess.run(
                    [total_loss,accuracy,softmax,train_step],
                    feed_dict={
                        x : color_data[train_idx[i*batch:i*batch+batch]],
                        y : labels[train_idx[i*batch:i*batch+batch]]
                        }
                )
            
            count = (j * (DATA_NUM-TEST_NUM)//batch) + i
            print(count,loss,acc)
        
    val = sess.run(var_list)
    with open(NUMPY_WEIGHT,'wb') as fid:
        cPickle.dump(val,fid,protocol=cPickle.HIGHEST_PROTOCOL)
    
    saver.save(sess, os.path.join(SAVE_DIR,MODEL_NAME)) 










