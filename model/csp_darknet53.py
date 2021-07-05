#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[ ]:


def conv2d_bn_mish(x, filters, kernel_size, strides=(1,1), padding='same',name=None):
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding, use_bias=False, name=name+"_conv2d")(x)
    x = tf.keras.layers.BatchNormalization(name=name+"_batch_normalization")(x)
    return x * tf.math.tanh(tf.math.softplus(x))


# In[ ]:


def scaled_yolov4_csp_block(x, filters, num_block = 3, type="backbone",name=None):

    right_branch_index = 0
    new_filters = filters
    if type == 'backbone':
        new_filters = filters // 2
    elif type == 'head':
        x = conv2d_bn_mish(x, new_filters, (1, 1), name=name + "_head")

    x_branch = tf.keras.layers.Conv2D(new_filters, (1, 1), 1, padding='same', use_bias=False,name=name+"_left_branch_conv2d")(x)
    if type == 'spp':
        x = conv2d_bn_mish(x, new_filters, (1, 1),name=name+"_right_branch_{}".format(right_branch_index))
        right_branch_index += 1
        x = conv2d_bn_mish(x, new_filters, (3, 3),name=name+"_right_branch_{}".format(right_branch_index))
        right_branch_index += 1
        x = conv2d_bn_mish(x, new_filters, (1, 1),name=name+"_right_branch_{}".format(right_branch_index))
        right_branch_index += 1
        pool_sizes = [5, 9, 13]
        pooling_results = [tf.keras.layers.MaxPooling2D((pool_size, pool_size), strides=(1, 1), padding='same')(x) for
                           pool_size in pool_sizes]
        x = tf.keras.layers.Concatenate()(pooling_results + [x])
        x = conv2d_bn_mish(x, new_filters, (1, 1),name=name+"_right_branch_{}".format(right_branch_index))
        right_branch_index += 1
        x = conv2d_bn_mish(x, new_filters, (3, 3),name=name+"_right_branch_{}".format(right_branch_index))
        right_branch_index += 1
        pass
    else:
        if type == 'backbone':
            x = conv2d_bn_mish(x, new_filters, (1, 1),name=name+"_right_branch_{}".format(right_branch_index))
            right_branch_index += 1

        for i in range(num_block):
            x1 = conv2d_bn_mish(x, new_filters, (1, 1),name=name+"_right_branch_res_{}".format(i*2))
            x1 = conv2d_bn_mish(x1, new_filters, (3, 3),name=name+"_right_branch_res_{}".format(i*2+1))
            if type == 'backbone':
                x = tf.keras.layers.Add()([x, x1])
            else:
                x = x1
        if type == 'backbone':
            x = tf.keras.layers.Conv2D(new_filters, (1, 1), 1, padding='same', use_bias=False,name=name+"_right_branch_{}_conv2d".format(right_branch_index))(x)
            right_branch_index += 1
    x = tf.keras.layers.Concatenate()([x, x_branch])
    x = tf.keras.layers.BatchNormalization(name=name+"_concat_batch_normalization")(x)
    x = x * tf.math.tanh(tf.math.softplus(x))
    return conv2d_bn_mish(x, filters, (1, 1),name =name + "_foot")


# In[ ]:


def csp_darknet_block(x, loop_num, filters, is_half_filters=True):

    x = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = conv2d_bn_mish(x, filters, (3,3), strides=(2,2), padding='valid')
    csp_branch = conv2d_bn_mish(x, filters//2 if is_half_filters else filters, (1, 1))
    darknet_branch = conv2d_bn_mish(x, filters//2 if is_half_filters else filters, (1, 1))

    for i in range(loop_num):
        x = conv2d_bn_mish(darknet_branch, filters//2, (1, 1))
        x = conv2d_bn_mish(x, filters // 2 if is_half_filters else filters, (3, 3))
        darknet_branch = tf.keras.layers.Add()([darknet_branch, x])

    darknet_branch = conv2d_bn_mish(darknet_branch, filters // 2 if is_half_filters else filters, (1, 1))

    x = tf.keras.layers.Concatenate()([darknet_branch, csp_branch])

    return conv2d_bn_mish(x, filters, (1, 1))


# In[ ]:


def scaled_yolov4_csp_darknet53(x,mode='p5'):

    darknet53_filters = [64 * 2 ** i for i in range(5)]
    if mode == 'p5':
        loop_nums = [1, 3, 15, 15, 7]
    elif mode == 'p6':
        loop_nums = [1, 3, 15, 15, 7, 7]
        darknet53_filters += [1024]
    elif mode == 'p7':
        loop_nums = [1, 3, 15, 15, 7, 7, 7]
        darknet53_filters += [1024]*2

    x = conv2d_bn_mish(x, 32, (3, 3), name="first_block")
    output_layers = []

    for block_index in range(len(loop_nums)):
        x = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
        x = conv2d_bn_mish(x, darknet53_filters[block_index], (3, 3), strides=(2, 2), padding='valid',name="backbone_block_{}_0".format(block_index))
        x = scaled_yolov4_csp_block(x, darknet53_filters[block_index],loop_nums[block_index], type="backbone",name="backbone_block_{}_1".format(block_index))
        output_layers.append(x)

    return output_layers[2:]

