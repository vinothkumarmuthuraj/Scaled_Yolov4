#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import logging
import time
import warnings
import numpy as np
import cv2
import tensorflow as tf


# In[ ]:


def random_left_right_flip(img, boxes):
#     print('lr')
    boxes1 = np.copy(boxes)
    img = np.fliplr(img)
    boxes1[..., [2,0]] = img.shape[1]  - boxes1[..., [0, 2]]
    return img,boxes1


# In[ ]:


def random_top_bottom_flip(img, boxes):
#     print('tp')
    boxes1 = np.copy(boxes)
    img = np.flipud(img)
    boxes1[..., [3,1]] = img.shape[1]  - boxes1[..., [1, 3]]
    return img,boxes1


# In[ ]:


def random_affine(img, targets, degrees=10, translate=0.1, scale=0.1, shear=10, border=0):

    height = img.shape[0] + border * 2
    width = img.shape[1] + border * 2
    
    img = np.copy(img)
    targets = np.copy(targets)


    # Rotation and Scale
    R = np.eye(3)
    a = np.random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = np.random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = np.random.uniform(-translate, translate) * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = np.random.uniform(-translate, translate) * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = np.math.tan(np.random.uniform(-shear, shear) * np.math.pi / 180)  # x shear (deg)
    S[1, 0] = np.math.tan(np.random.uniform(-shear, shear) * np.math.pi / 180)  # y shear (deg)

    # Combined rotation matrix
    M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
    if (border != 0) or (M != np.eye(3)).any():  # image changed
        img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(targets)

    
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        
        xy[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = (xy @ M.T)[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # reject warped points outside of image
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]
        area = w * h
        area0 = (targets[:, 2] - targets[:, 0]) * (targets[:, 3] - targets[:, 1])
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
        i = (w > 4) & (h > 4) & (area / (area0 * s + 1e-16) > 0.2) & (ar < 10)
        
        for j,k in enumerate(i):
            if k==True:
                targets[j, 0:4] = xy[j]
            elif k==False:
                targets[j,0:4] = 0

    return img, targets


# In[ ]:


def random_color_distort(image, data_rng=None, eig_val=None,
                            eig_vec=None, var=5, alphastd=0.1):


    if data_rng is None:
        data_rng = np.random.RandomState(None)
    if eig_val is None:
        eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                           dtype=np.float32)
    if eig_vec is None:
        eig_vec = np.array([[-0.58752847, -0.69563484, 0.41340352],
                            [-0.5832747, 0.00994535, -0.81221408],
                            [-0.56089297, 0.71832671, 0.41158938]], dtype=np.float32)

    def grayscale(image):
        return tf.image.rgb_to_grayscale(image)

    def lighting_(data_rng, image, alphastd, eigval, eigvec):
        alpha = data_rng.normal(scale=alphastd, size=(3, ))
        image += np.dot(eigvec, eigval * alpha)
        return image

    def blend_(alpha, image1, image2):

        image1 *= alpha
        image2 *= (1 - alpha)
        image1 += image2
        return image1


    def saturation_(data_rng, image, gs, gs_mean, var):
        # pylint: disable=unused-argument
        alpha = 1. + data_rng.uniform(low=-var, high=var)
        image = blend_(alpha, image, gs)
        return image

    def brightness_(data_rng, image, gs, gs_mean, var):
        # pylint: disable=unused-argument
        alpha = 1. + data_rng.uniform(low=-var, high=var)
        image *= alpha
        return image


    def contrast_(data_rng, image, gs, gs_mean, var):
        # pylint: disable=unused-argument
        alpha = 1. + data_rng.uniform(low=-var, high=var)
        image = blend_(alpha, image, gs_mean)
        return image

    functions = [saturation_,brightness_,contrast_,lighting_]
    fun_exc = np.random.choice(functions)

    gs = grayscale(image)
    gs_mean = tf.math.reduce_mean(gs)
    
    if fun_exc == lighting_:
        image = fun_exc(data_rng, image, alphastd, eig_val, eig_vec)
    elif fun_exc == saturation_ or brightness_ or contrast_:
        image = fun_exc(data_rng, image, gs, gs_mean, var)

    return image

