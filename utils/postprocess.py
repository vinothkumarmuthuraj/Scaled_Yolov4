#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf


# In[2]:


from utils.box_decode import box_decode


# In[ ]:


def postprocess(outputs,args):

    boxes_list = []
    scores_list = []
    for index, output in enumerate(outputs):

        output = tf.sigmoid(output)
        decoded_boxes = box_decode(output[..., 0:4], index, args)

        scores = output[..., 4:5] * output[..., 5:]


        scores = tf.reshape(scores, [tf.shape(scores)[0], -1, tf.shape(scores)[-1]])

        boxes_list.append(decoded_boxes)
        scores_list.append(scores)
    decoded_boxes = tf.concat(boxes_list, axis=-2,name='output_boxes')
    scores = tf.concat(scores_list, axis=-2,name='output_scores')
    return decoded_boxes, scores

