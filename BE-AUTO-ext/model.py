# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 18:11:00 2019

@author: pcm
"""
import tensorflow as tf
#import tensorlayer as tl
from tensorlayer.layers import (Input, DeConv2d, BatchNorm, Elementwise,Conv2d,Concat)
from tensorlayer.models import Model
def get_G(input_tensor):
    with tf.device('/gpu:0'):
        
        ##exp1:conv-enc-deconv-dec
        w_init = tf.random_normal_initializer(stddev=0.02)
        g_init = tf.random_normal_initializer(1., 0.02)
        
        nin = Input(input_tensor)
        c1 = Conv2d(16, (3, 3), (1, 1), act=tf.nn.relu,padding='SAME', W_init=w_init)(nin)
        c2 = Conv2d(32, (3, 3), (1, 1), act=tf.nn.relu,padding='SAME', W_init=w_init)(c1)
        c3 = Conv2d(64, (3, 3), (1, 1), act=tf.nn.relu,padding='SAME', W_init=w_init)(c2)
        c4 = Conv2d(128, (3, 3), (1, 1), act=tf.nn.relu,padding='SAME', W_init=w_init)(c3)
        c5 = Conv2d(256, (3, 3), (1, 1), act=tf.nn.relu,padding='SAME', W_init=w_init)(c4)
        
        d4=  DeConv2d(128, (3, 3), (1, 1),padding='SAME', W_init=w_init)(c5)
        d4=  BatchNorm(act=tf.nn.relu,gamma_init=g_init)(d4)
        d4=  Elementwise(tf.add)([c4, d4])
        d3=  DeConv2d(64, (3, 3), (1, 1), padding='SAME', W_init=w_init)(d4)
        d3=  BatchNorm(act=tf.nn.relu,gamma_init=g_init)(d3)
        d3=  Elementwise(tf.add)([c3, d3])
        d2=  DeConv2d(32, (3, 3), (1, 1),padding='SAME', W_init=w_init)(d3)
        d2=  BatchNorm(act=tf.nn.relu,gamma_init=g_init)(d2)
        d2=  Elementwise(tf.add)([c2, d2])
        d1=  DeConv2d(16, (3, 3), (1, 1),padding='SAME', W_init=w_init)(d2)
        d1=  BatchNorm(act=tf.nn.relu,gamma_init=g_init)(d1)
        d1=  Elementwise(tf.add)([c1, d1])
        out=  DeConv2d(3, (3, 3), (1, 1),padding='SAME', W_init=w_init)(d1)
        out=  BatchNorm(act=tf.nn.relu,gamma_init=g_init)(out)
        out=  Elementwise(tf.add)([out, nin])
        G = Model(inputs=nin, outputs=out, name="generator")
        return G
        
        #exp2:full conv enc-dec
#        w_init = tf.random_normal_initializer(stddev=0.02)
#        g_init = tf.random_normal_initializer(1., 0.02)
        
#        nin = Input(input_tensor)
#        c1 = Conv2d(16, (3, 3), (1, 1), act=tf.nn.relu,padding='SAME', W_init=w_init)(nin)
#        c2 = Conv2d(32, (3, 3), (1, 1), act=tf.nn.relu,padding='SAME', W_init=w_init)(c1)
#        c3 = Conv2d(64, (3, 3), (1, 1), act=tf.nn.relu,padding='SAME', W_init=w_init)(c2)
#        c4 = Conv2d(128, (3, 3), (1, 1), act=tf.nn.relu,padding='SAME', W_init=w_init)(c3)
#        c5 = Conv2d(256, (3, 3), (1, 1), act=tf.nn.relu,padding='SAME', W_init=w_init)(c4)
        
#        d4=  Conv2d(128, (3, 3), (1, 1),padding='SAME', W_init=w_init)(c5)
#        d4=  BatchNorm(act=tf.nn.relu,gamma_init=g_init)(d4)
#        d4=  Elementwise(tf.add)([c4, d4])
#        d3=  Conv2d(64, (3, 3), (1, 1), padding='SAME', W_init=w_init)(d4)
#        d3=  BatchNorm(act=tf.nn.relu,gamma_init=g_init)(d3)
#        d3=  Elementwise(tf.add)([c3, d3])
#        d2=  Conv2d(32, (3, 3), (1, 1),padding='SAME', W_init=w_init)(d3)
#        d2=  BatchNorm(act=tf.nn.relu,gamma_init=g_init)(d2)
#        d2=  Elementwise(tf.add)([c2, d2])
#        d1=  Conv2d(16, (3, 3), (1, 1),padding='SAME', W_init=w_init)(d2)
#        d1=  BatchNorm(act=tf.nn.relu,gamma_init=g_init)(d1)
#        d1=  Elementwise(tf.add)([c1, d1])
#        out=  Conv2d(3, (3, 3), (1, 1),padding='SAME', W_init=w_init)(d1)
#        out=  BatchNorm(act=tf.nn.relu,gamma_init=g_init)(out)
#        out=  Elementwise(tf.add)([out, nin])
#        G = Model(inputs=nin, outputs=out, name="generator")
#        return G
        
#        ##exp5:full deconv
#        w_init = tf.random_normal_initializer(stddev=0.02)
#        g_init = tf.random_normal_initializer(1., 0.02)
#        
#        nin = Input(input_tensor)
#        c1 = DeConv2d(16, (3, 3), (1, 1), act=tf.nn.relu,padding='SAME', W_init=w_init)(nin)
#        c2 = DeConv2d(32, (3, 3), (1, 1), act=tf.nn.relu,padding='SAME', W_init=w_init)(c1)
#        c3 = DeConv2d(64, (3, 3), (1, 1), act=tf.nn.relu,padding='SAME', W_init=w_init)(c2)
#        c4 = DeConv2d(128, (3, 3), (1, 1), act=tf.nn.relu,padding='SAME', W_init=w_init)(c3)
#        c5 = DeConv2d(256, (3, 3), (1, 1), act=tf.nn.relu,padding='SAME', W_init=w_init)(c4)
#        
#        d4=  DeConv2d(128, (3, 3), (1, 1),padding='SAME', W_init=w_init)(c5)
#        d4=  BatchNorm(act=tf.nn.relu,gamma_init=g_init)(d4)
#        d4=  Elementwise(tf.add)([c4, d4])
#        d3=  DeConv2d(64, (3, 3), (1, 1), padding='SAME', W_init=w_init)(d4)
#        d3=  BatchNorm(act=tf.nn.relu,gamma_init=g_init)(d3)
#        d3=  Elementwise(tf.add)([c3, d3])
#        d2=  DeConv2d(32, (3, 3), (1, 1),padding='SAME', W_init=w_init)(d3)
#        d2=  BatchNorm(act=tf.nn.relu,gamma_init=g_init)(d2)
#        d2=  Elementwise(tf.add)([c2, d2])
#        d1=  DeConv2d(16, (3, 3), (1, 1),padding='SAME', W_init=w_init)(d2)
#        d1=  BatchNorm(act=tf.nn.relu,gamma_init=g_init)(d1)
#        d1=  Elementwise(tf.add)([c1, d1])
#        out=  DeConv2d(3, (3, 3), (1, 1),padding='SAME', W_init=w_init)(d1)
#        out=  BatchNorm(act=tf.nn.relu,gamma_init=g_init)(out)
#        out=  Elementwise(tf.add)([out, nin])
#        G = Model(inputs=nin, outputs=out, name="generator")
#        return G
        
