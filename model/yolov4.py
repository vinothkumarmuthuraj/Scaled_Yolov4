#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[ ]:


from model.csp_darknet53 import scaled_yolov4_csp_darknet53
from model.head import head, yolov3_head


# In[ ]:


def Yolov4(args, training=True):
    if args.fixed_scale_bool == True:
        input = tf.keras.layers.Input((args.fixed_scale, args.fixed_scale, 3))
    else:
        input = tf.keras.layers.Input((None, None, 3))
    scaled_yolov4_csp_darknet53_outputs = scaled_yolov4_csp_darknet53(input,mode=args.model_type)
    head_outputs = head(args,scaled_yolov4_csp_darknet53_outputs)
    outputs = yolov3_head(head_outputs, args)

    model = tf.keras.Model(inputs=input, outputs=outputs)
    return model

