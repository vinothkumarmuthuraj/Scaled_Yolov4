#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow as tf
import os
import xml.etree.ElementTree as ET
import copy


from generator.data_augment import random_left_right_flip,random_top_bottom_flip,random_affine,random_color_distort

from generator.preprocess_true_boxes import boxes_iou,process_true_boxes

from utils.anchors import yolo_anchors


# In[ ]:


class VocGenerator():
    def __init__(self, args, mode=0):
        
        self._args = args
        self.mode = mode
        
        if self.mode == 0:
            self.ann_dir = self._args.train_annot_directory
            self.img_dir = self._args.train_image_directory
            self.batch_size = self._args.train_batch_size
        else:
            self.ann_dir = self._args.valid_annot_directory
            self.img_dir = self._args.valid_image_directory  
            self.batch_size = self._args.valid_batch_size
            
        f = open(self._args.class_names)
        self.labels = f.read().splitlines()
        
        self.fixed_scale_bool = self._args.fixed_scale_bool
        self.fixed_scale = self._args.fixed_scale
        self.multi_scale = self._args.multi_scale
        self.anchors = yolo_anchors[self._args.model_type]
        self.num_class = len(self.labels)
        self.anchor_match_iou_thr = self._args.anchor_match_iou_thr
        self.offset = self._args.offset
        
        self.data_augment = self._args.data_augment
        
        if self._args.model_type == 'tiny':
            detect_layer_num = self.anchors.shape[0]
            self.anchor_per_scale = self.anchors.shape[1]
            self.strides = [16 * 2 ** i for i in range(detect_layer_num)]
            
        if self._args.model_type == 'p5':
            detect_layer_num = self.anchors.shape[0]
            self.anchor_per_scale = self.anchors.shape[1]
            self.strides = [8 * 2 ** i for i in range(detect_layer_num)]
            
        if self._args.model_type == 'p6':
            detect_layer_num = self.anchors.shape[0]
            self.anchor_per_scale = self.anchors.shape[1]
            self.strides = [8 * 2 ** i for i in range(detect_layer_num)]
            
        if self._args.model_type == 'p7':
            detect_layer_num = self.anchors.shape[0]
            self.anchor_per_scale = self.anchors.shape[1]
            self.strides = [8 * 2 ** i for i in range(detect_layer_num)]
        
        
    def parse_annotation(self,ann_dir,img_dir,labels):
    
        max_annot = 0
        imgs_name = []
        annots = []
    
        # Parse file
        for ann in sorted(os.listdir(ann_dir)):
            annot_count = 0
            boxes = []

            tree = ET.parse(ann_dir +'\\'+ ann)
            for elem in tree.iter(): 
                if 'filename' in elem.tag:
                    imgs_name.append(img_dir +'\\'+ elem.text)
                    image_path=img_dir + '\\' + elem.text
                if 'width' in elem.tag:
                    w = int(elem.text)
                if 'height' in elem.tag:
                    h = int(elem.text)
                if 'object' in elem.tag or 'part' in elem.tag:                  
                    box = np.zeros((5))
                    for attr in list(elem):
                        if 'name' in attr.tag:
                            box[4] = labels.index(attr.text)+1  # 0:label for no bounding box
                        if 'bndbox' in attr.tag:
                            annot_count += 1

                            for dim in list(attr):
                                if 'xmin' in dim.tag:
                                    box[0] = int(round(float(dim.text)))
                                if 'ymin' in dim.tag:
                                    box[1] = int(round(float(dim.text)))
                                if 'xmax' in dim.tag:
                                    box[2] = int(round(float(dim.text)))
                                if 'ymax' in dim.tag:
                                    box[3] = int(round(float(dim.text)))

                    boxes.append(np.asarray(box))
   
            annots.append(np.asarray(boxes))
        

            if annot_count > max_annot:
                max_annot = annot_count
  
    # Rectify annotations boxes : len -> max_annot
        imgs_name = np.array(imgs_name)  
        true_boxes = np.zeros((imgs_name.shape[0], max_annot, 5))

        for idx, boxes in enumerate(annots):

            true_boxes[idx, :boxes.shape[0], :5] = boxes
        
        return imgs_name,true_boxes
        
    def parse_function(self,img_obj, true_boxes):
        
        if self.fixed_scale_bool == True:
            random_img_size = self.fixed_scale
        else:
            random_img_size = np.random.choice(self.multi_scale)
            
        ih= iw = random_img_size
    
        x_img_string = tf.io.read_file(img_obj)
        x_img = tf.io.decode_jpeg(x_img_string, channels=3)
        x_img = tf.image.convert_image_dtype(x_img, tf.float32)
    
    
        h  = tf.cast(tf.shape(x_img)[0],dtype = tf.float32)
        w  = tf.cast(tf.shape(x_img)[1],dtype = tf.float32)

        scale = tf.math.reduce_min([tf.math.divide(iw,w), tf.math.divide(ih,h)])
        nw, nh  = tf.math.multiply(scale, w), tf.math.multiply(scale, h)
        image_resized = tf.image.resize_with_pad(image = x_img,target_height = ih,target_width = iw)
        dw, dh = (iw - nw) // 2, (ih-nh) // 2
    
        true_boxes = tf.cast(true_boxes,dtype=tf.float32)
    
        if true_boxes is None:
            return image_resized
    
        else:
            true_box_row = tf.shape(true_boxes)[0]
            true_box_col = tf.shape(true_boxes)[1]
            x1 = true_boxes[..., 0] * scale + dw
            y1 = true_boxes[..., 1] * scale + dh
            x2 = true_boxes[..., 2] * scale + dw
            y2 = true_boxes[..., 3] * scale + dh
            class1 = true_boxes[...,4]
            concat = tf.concat([x1,y1,x2,y2,class1],axis=-1)
            reshape = tf.reshape(concat,[true_box_col,true_box_row])
            transpose_box = tf.transpose(reshape,perm=[1,0])
            return image_resized, transpose_box
        
    @tf.autograph.experimental.do_not_convert
    def get_dataset(self,img_dir, ann_dir, labels, batch_size):

        images, true_boxes = self.parse_annotation(ann_dir, img_dir, labels)
    
        dataset = tf.data.Dataset.from_tensor_slices((images, true_boxes))    
        dataset = dataset.shuffle(len(images))
        dataset = dataset.repeat()
        dataset = dataset.map(self.parse_function, num_parallel_calls=6)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(10)
#         print('-------------------')
#         print('Dataset:')
#         print('Images count: {}'.format(len(images)))
#         print('Step per epoch: {}'.format(len(images) // batch_size))
#         print('Images per epoch: {}'.format(batch_size * (len(images) // batch_size)))
        return dataset
    
    def augmentation_generator(self,train_dataset,data_augment):
        fun_exc = np.random.choice(data_augment)
    
        for batch in train_dataset:
            image = batch[0].numpy()
            boxes = batch[1]. numpy()

            for ind in range(batch[0].shape[0]):
                if str(fun_exc) == 'lrtp':
                    function = [random_left_right_flip,random_top_bottom_flip]
                    random_number = [np.random.random() for i in range(len(function))]
                    random_number_arr = np.array(random_number)
                    if np.any(random_number_arr<0.5):
                        for ran in random_number:
                            if ran<0.50:
                                image_aug,boxes_aug = function[random_number.index(ran)](image[ind],boxes[ind])
                                image[ind] = image_aug
                                boxes[ind] = boxes_aug

                elif str(fun_exc) == 'random_affine':
#                     print('affine')
                    image_aug,boxes_aug = random_affine(image[ind],boxes[ind])
                    image[ind] = image_aug
                    boxes[ind] = boxes_aug
            
                elif str(fun_exc) == 'random_color_distort':
                    image_aug = random_color_distort(image[ind])
                    image[ind] = image_aug
                
                elif str(fun_exc) == 'None':
                    continue

            batch = (tf.convert_to_tensor(image), tf.convert_to_tensor(boxes))
            yield batch
            
    def ground_truth_generator(self,dataset):
    
        for batch in dataset:
        
            imgs = batch[0]
        
            true_boxes = batch[1].numpy() 
            batch_size,img_h,img_w,_ = imgs.shape
            max_side = np.max([img_h,img_w])
            Train_output_sizes = max_side//np.array(self.strides)        
            true_boxes[..., 0:4] /= np.tile([img_h,img_w],[2])   

        
            y_true = process_true_boxes(true_boxes,Train_output_sizes,max_side,self.anchors,self.num_class,
                                        self.anchor_per_scale,self.anchor_match_iou_thr,self.offset)

            yield (imgs,y_true,true_boxes)
            
    def __iter__(self):
        return self
    
    def __next__(self):
        dataset = self.get_dataset(self.img_dir,self.ann_dir,self.labels,self.batch_size)
        
        if self.mode == 0: 
            aug_train_dataset = self.augmentation_generator(dataset,self.data_augment)
            train_gen = self.ground_truth_generator(aug_train_dataset)
            imgs,y_true,true_boxes = next(train_gen)
        else:
            valid_gen = self.ground_truth_generator(dataset)
            imgs,y_true,true_boxes = next(valid_gen)
            
        return imgs,y_true,true_boxes

