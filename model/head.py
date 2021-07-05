#!/usr/bin/env python
# coding: utf-8

# In[6]:


import tensorflow as tf
import numpy as np


# In[10]:


from model.csp_darknet53 import conv2d_bn_mish, scaled_yolov4_csp_block
from utils.anchors import yolo_anchors


# In[ ]:


def head_down(args,backbone_output_layers):
    output_layers = []
    head_down_output_layer = scaled_yolov4_csp_block(backbone_output_layers[-1],backbone_output_layers[-1].shape[-1]//2,3,type='spp',name='spp')
    output_layers.append(head_down_output_layer)
    head_down_block_index = 0
    for layer_index in range(2,len(backbone_output_layers)+1):
        if layer_index<len(backbone_output_layers)-1:
            spp_out_channel = head_down_output_layer.shape[-1]
        else:
            spp_out_channel = head_down_output_layer.shape[-1] // 2
        head_down_output_layer = conv2d_bn_mish(head_down_output_layer,spp_out_channel ,(1, 1),
                                                name="head_down_block_{}_0".format(head_down_block_index))
        head_down_output_layer = tf.keras.layers.UpSampling2D(size=(2, 2))(head_down_output_layer)

        backbone_output_layer = conv2d_bn_mish(backbone_output_layers[-layer_index], backbone_output_layers[-layer_index].shape[3] // 2,(1, 1),
                                               name="head_down_block_{}_1".format(head_down_block_index))
        if args.fixed_scale_bool == True:
            if backbone_output_layer.shape[1]%2 != 0:
                if backbone_output_layer.shape[1] != head_down_output_layer.shape[1]:
                    head_down_output_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0))) (head_down_output_layer)
        
        head_down_output_layer = tf.keras.layers.Concatenate()([backbone_output_layer, head_down_output_layer])
        head_down_output_layer = scaled_yolov4_csp_block(head_down_output_layer,head_down_output_layer.shape[-1] // 2, 3, type='head',
                                                        name="head_down_block_{}_2".format(head_down_block_index))
        head_down_block_index += 1
        output_layers.append(head_down_output_layer)
    return output_layers


# In[ ]:


def head_up(head_down_output_layers):
    output_layers = []
    head_up_output_layers = head_down_output_layers[-1]
    output_layers.append(head_up_output_layers)
    head_up_block_index = 0

    for layer_index in range(2, len(head_down_output_layers) + 1):
        if layer_index>=4:
            head_up_out_channel = head_up_output_layers.shape[-1]
        else:
            head_up_out_channel = head_up_output_layers.shape[-1] *2

        head_up_output_layers = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(head_up_output_layers)
        head_up_output_layers = conv2d_bn_mish(head_up_output_layers,head_up_out_channel , (3,3), strides=(2,2), padding='valid',
                                               name="head_up_block_{}_0".format(head_up_block_index))
        head_up_output_layers = tf.keras.layers.Concatenate()([head_up_output_layers, head_down_output_layers[-layer_index]])
        head_up_output_layers = scaled_yolov4_csp_block(head_up_output_layers, head_up_output_layers.shape[-1] // 2, 3, type='head',
                                                         name="head_up_block_{}_1".format(head_up_block_index))
        head_up_block_index += 1
        output_layers.append(head_up_output_layers)
    return output_layers


# In[ ]:


def head(args,x):
    x = head_down(args,x)
    x = head_up(x)
    return x


# In[ ]:


def yolov3_head(inputs, args):

    f = open(args.class_names)
    labels = f.read().splitlines()
    class_num = len(labels)
   
    anchors = yolo_anchors[args.model_type]
    output_layers = []
    for index, x in enumerate(inputs):
        x = conv2d_bn_mish(x, x.shape[-1]*2, (3, 3), name='yolov3_head_%d_0'%index)
        x = tf.keras.layers.Conv2D(len(anchors[index])*(class_num+5), (1, 1), use_bias=True, name='yolov3_head_%d_1_conv2d'%index)(x)
        x = tf.reshape(x,[tf.shape(x)[0], tf.shape(x)[1],tf.shape(x)[2],-1,class_num+5])
        output_layers.append(x)

    return output_layers

