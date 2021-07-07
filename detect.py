#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import cv2
import copy
import math
import glob
import time
import argparse
import colorsys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# In[2]:


from model.yolov4 import Yolov4
from model.yolov4_tiny import Yolov4_tiny

from utils.anchors import yolo_anchors

from utils.postprocess import postprocess
from utils.nms import yolov4_nms


# In[ ]:


def parse_args(args):
    parser = argparse.ArgumentParser(description='Simple detection script for using ScaledYOLOv4.')
    parser.add_argument('--model_type', default='tiny',help="choices=['tiny','p5','p6','p7']")
    parser.add_argument('--image_directory', default='./image_directory')
    parser.add_argument('--img_save_path', default='./img_save_path')
    parser.add_argument('--checkpoint_dir', default='./checkpoint_dir')
    parser.add_argument('--class_names', default='class_names.names',help="voc.names")
    parser.add_argument('--optimizer', default='Adam', help="choices=[Adam,sgd]")
    parser.add_argument('--fixed_scale_bool', default='False', type=bool,help="choices=[True,False]")
    parser.add_argument('--fixed_scale', default=608, type=int)
    parser.add_argument('--detect_img_size', default=608, type=int)
    parser.add_argument('--drop_block', default='False', help="choices=[True,False]")
    parser.add_argument('--scales_x_y', default=[2., 2., 2., 2., 2.])
    parser.add_argument('--nms', default='diou_nms', help="choices=['diou_nms','hard_nms']")
    parser.add_argument('--nms_max_box_num', default=100, type=int)
    parser.add_argument('--nms_iou_threshold', default=0.20, type=float)
    parser.add_argument('--nms_score_threshold', default=0.25, type=float)
    
    return parser.parse_args(args)


# In[8]:


def detect_batch_img(img,model,args):

    pred = model.predict(img)
    pre_nms_decoded_boxes,pre_nms__scores = postprocess(pred,args)
    pre_nms_decoded_boxes = pre_nms_decoded_boxes.numpy()
    pre_nms__scores = pre_nms__scores.numpy()
    boxes, scores, classes, valid_detections = yolov4_nms(args)(pre_nms_decoded_boxes, pre_nms__scores, args)

    return boxes, scores, classes, valid_detections


# In[9]:


def main(args):
    
    f = open(args.class_names)
    labels = f.read().splitlines()
    class_num = len(labels)
    
    if args.model_type == "tiny":
        print('tiny activated')
        model = Yolov4_tiny(args, training=True)
        
    elif args.model_type == "p5":
        model = Yolov4(args, training=True)
        
    elif args.model_type == "p6":
        model = Yolov4(args, training=True)
        
    else:
        model = Yolov4(args, training=True)
    
    # optimizer
    if args.optimizer == 'Adam':
        optimizer = tf.keras.optimizers.Adam()
    elif args.optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD()
        
    root = tf.train.Checkpoint(optimizer=optimizer,model=model)
    manager = tf.train.CheckpointManager(root,args.checkpoint_dir, max_to_keep=3)
    print(manager.checkpoints)
    root.restore(manager.latest_checkpoint)
    
    hsv_tuples = [(1.0 * x / class_num, 1., 1.) for x in range(class_num)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255.), int(x[1] * 255.), int(x[2] * 255.)), colors))
    
    files=glob.glob(args.image_directory + '\*.jpg')
#     print(files)
    for i in range(len(files)):
        start=time.time()
        base_name = os.path.basename(files[i])
        
        x_img_string = tf.io.read_file(files[i])
        original_image = tf.io.decode_jpeg(x_img_string, channels=3)
        original_image = tf.image.convert_image_dtype(original_image, tf.float32)
        
        if args.fixed_scale_bool == True:
            image_resize = tf.image.resize_with_pad(image = original_image,target_height = args.fixed_scale,
                                                     target_width = args.fixed_scale)
        else:
            image_resize = tf.image.resize_with_pad(image = original_image,target_height = args.detect_img_size,
                                                     target_width = args.detect_img_size)
            
        image_resize_shape = image_resize.shape
        image_data =np.expand_dims(image_resize, axis=0)
        
        #converted to numpy for the purpose of plotting boxes
        image_resize_ny = image_resize.numpy()*255
        
        boxes,scores,classes,valid_detections = detect_batch_img(image_data, model,args)

        batch_index = 0
        valid_boxes = (boxes[batch_index][0:valid_detections[batch_index]]) * image_resize_shape[0]
        valid_classes = classes[batch_index][0:valid_detections[batch_index]]
        valid_scores = scores[batch_index][0:valid_detections[batch_index]]
        
        count_detected = valid_boxes.shape[0]
        for i in range(count_detected):
            box = valid_boxes[i][:4]
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
            valid_class = valid_classes[i]
            
            fontScale = 0.5
            score = valid_scores[i]
            bbox_color = colors[valid_class]
            bbox_thick = 2
            c1, c2 = (x1, y1), (x2, y2)
            
            cv2.rectangle(image_resize_ny, (x1,y1) ,(x2,y2), bbox_color, bbox_thick)
            
            bbox_mess = '%s: %.2f' % (labels[valid_class], score)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
            c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
            cv2.rectangle(image_resize_ny, c1, (np.float32(c3[0]), np.float32(c3[1])), bbox_color, -1) #filled

            cv2.putText(image_resize_ny, bbox_mess, (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
            
        tf.keras.preprocessing.image.save_img(args.img_save_path + '\{}'.format(base_name),image_resize_ny)
        plt.imshow(image_resize_ny)  
        end=time.time()
        print(f"Runtime of the program is {end - start}")


# In[10]:


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)

