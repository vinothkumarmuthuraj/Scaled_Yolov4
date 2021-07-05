#!/usr/bin/env python
# coding: utf-8

# In[6]:


import tensorflow as tf
import numpy as np


# In[7]:


from utils.anchors import yolo_anchors
from model.drop_block import DropBlock2D


# In[20]:


def conv2d_bn_leaky(x, filters, kernel_size, strides=(1,1), padding='same',name=None):
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding, use_bias=False, name=name+"_conv2d")(x)
    x = tf.keras.layers.BatchNormalization(name=name+"_batch_normalization")(x)
    return tf.keras.layers.LeakyReLU(alpha=0.1)(x)


# In[21]:


def tiny_block(x,name):
    x = conv2d_bn_leaky(x, x.shape[-1],(3, 3),name=name+"_1")
    x1 = x[..., x.shape[-1]//2:]
    x2 = conv2d_bn_leaky(x1, x1.shape[-1], (3, 3),name=name+"_2")
    x3 = conv2d_bn_leaky(x2, x2.shape[-1], (3, 3),name=name+"_3")
    x3 = tf.keras.layers.Concatenate()([x3,x2])
    x3 = conv2d_bn_leaky(x3, x3.shape[-1], (1, 1),name=name+"_4")
    x4 = tf.keras.layers.Concatenate()([x, x3])
    return x4,x3


# In[22]:


def backbone(x,args):
    x = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = conv2d_bn_leaky(x,32,(3,3),strides=(2,2),padding='valid',name="block_1_1")
    x = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = conv2d_bn_leaky(x,64,(3,3),strides=(2,2),padding='valid',name="block_2_1")

    x,_ = tiny_block(x,name="block_3")
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    
    if args.drop_block == True:
        drop_block1 = DropBlock2D(keep_prob=0.98, block_size=2)
        x = drop_block1(x,training=True)
        
    x,_ = tiny_block(x,name="block_4")
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    
    if args.drop_block == True:
        drop_block2 = DropBlock2D(keep_prob=0.98, block_size=2)
        x = drop_block2(x,training=True)
        
    x,x1 = tiny_block(x,name="block_5")
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = conv2d_bn_leaky(x, x.shape[-1], (3, 3), strides=(1, 1), padding='same',name="block_5_5")
    output1 = conv2d_bn_leaky(x, x.shape[-1]//2, (1, 1), strides=(1, 1), padding='same',name="block_5_6")
    x = conv2d_bn_leaky(output1, output1.shape[-1] // 2, (1, 1), strides=(1, 1), padding='same',name="block_5_7")
    x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
    output2 = tf.keras.layers.Concatenate()([x, x1])

    return [output2, output1]


# In[23]:


def head(inputs, args):
    
    f = open(args.class_names)
    labels = f.read().splitlines()
    class_num = len(labels)

    anchors = yolo_anchors[args.model_type]
    output_layers = []
    head_conv_filters = [256, 512]

    for index, x in enumerate(inputs):

        x = conv2d_bn_leaky(x, head_conv_filters[index], (3, 3), name='yolov3_head_%d_1' % (index+1))
        x = tf.keras.layers.Conv2D(len(anchors[index]) * (class_num + 5), (1, 1), use_bias=True,
                                   name='yolov3_head_%d_2_conv2d' % (index+1))(x)
        x = tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], -1, class_num + 5])
        output_layers.append(x)
    return output_layers


# In[24]:


def Yolov4_tiny(args, training=True):
    if args.fixed_scale_bool == True:
        input = tf.keras.layers.Input((args.fixed_scale, args.fixed_scale, 3))
    else:
        input = tf.keras.layers.Input((None, None, 3))
    
    outputs = backbone(input,args)
    outputs = head(outputs,args)

    if training:
        model = tf.keras.Model(inputs=input, outputs=outputs)
        return model


# In[ ]:




