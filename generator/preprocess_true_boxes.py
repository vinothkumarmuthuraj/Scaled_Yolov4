#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np


# In[2]:


def boxes_iou(boxes1, boxes2):
    boxes2 = np.array(boxes2)
    boxes1 = np.expand_dims(boxes1, -2)
    boxes1_wh = boxes1[..., 2:4] - boxes1[..., 0:2]
    inter_area = np.minimum(boxes1_wh[..., 0], boxes2[..., 0]) * np.minimum(boxes1_wh[..., 1], boxes2[..., 1])
    boxes1_area = boxes1_wh[..., 0] * boxes1_wh[..., 1]
    boxes2_area = boxes2[..., 0] * boxes2[..., 1]
    return inter_area/(boxes1_area+boxes2_area-inter_area)


# In[ ]:


def process_true_boxes(true_boxes,train_output_sizes,max_side,anchors,num_classes,
                       anchor_per_scale,anchor_match_iou_thr,offset):
      
    max_bbox_per_scale=true_boxes.shape[1]
    batch_size = true_boxes.shape[0]
    batch_true_boxes = np.zeros(batch_size,)
   
    grids = [np.zeros((batch_size,train_output_sizes[i],train_output_sizes[i],
                       anchor_per_scale,5 + num_classes),dtype=np.float32)for i in range(len(train_output_sizes))]
    
    iou_scores = boxes_iou(true_boxes, np.reshape(anchors, [-1,2])/(max_side, max_side))
    matched_anchor_index = np.argsort(-iou_scores,axis=-1)
    matched_anchor_num = np.sum(iou_scores>anchor_match_iou_thr, axis=-1)
    matched_anchor_num = np.expand_dims(matched_anchor_num, axis=-1)
    batch_boxes = np.concatenate([true_boxes, matched_anchor_index,matched_anchor_num], axis=-1)

    for batch_index in range(batch_size):
        for box_index in range(batch_boxes[batch_index].shape[0]):
            bbox_class_ind = int(batch_boxes[batch_index][box_index][4])
            onehot = np.zeros(num_classes, dtype=np.float)
            onehot = np.zeros(num_classes+1, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            onehot1=onehot[1:]
            uniform_distribution = np.full(num_classes, 1.0 / num_classes)
            deta = 0.001
            smooth_onehot = onehot1 * (1 - deta) + deta * uniform_distribution
            
            if np.any(iou_scores[batch_index][box_index]):
                if int(batch_boxes[batch_index][box_index][-1]) != 0:
                    
                    for anchor_index in batch_boxes[batch_index][box_index][5:5 + int(batch_boxes[batch_index][box_index][-1])]:
                       
                        grid_index = (anchor_index // len(anchors[0])).astype(np.int32)
                        grid_anchor_index = (anchor_index % len(anchors[0])).astype(np.int32)
                        cxcy = (batch_boxes[batch_index][box_index][0:2]+batch_boxes[batch_index][box_index][2:4])/2
                        cxcy *= train_output_sizes[grid_index]

                        grid_xy = np.floor(cxcy).astype(np.int32)

                        dxdy = cxcy - grid_xy
                        dwdh = batch_boxes[batch_index][box_index][2:4]-batch_boxes[batch_index][box_index][0:2]
                        dwdh = dwdh*np.array(train_output_sizes[grid_index])
           
                        grids[grid_index][batch_index][grid_xy[1]][grid_xy[0]][grid_anchor_index][0:4] = np.concatenate([dxdy,dwdh])
                        grids[grid_index][batch_index][grid_xy[1]][grid_xy[0]][grid_anchor_index][4] = 1
                        grids[grid_index][batch_index][grid_xy[1]][grid_xy[0]][grid_anchor_index][5:] = smooth_onehot

                        grid_xy_fract = cxcy%1.
                    
                        if (grid_xy>0).all():
                            if grid_xy_fract[0] < offset:
                                dxdy = cxcy - np.floor(cxcy - [0.5, 0.])
                                grids[grid_index][batch_index][grid_xy[1]][grid_xy[0]-1][grid_anchor_index][0:4] = np.concatenate([dxdy,dwdh])
                                grids[grid_index][batch_index][grid_xy[1]][grid_xy[0]-1][grid_anchor_index][4] = 1
                                grids[grid_index][batch_index][grid_xy[1]][grid_xy[0]-1][grid_anchor_index][5:] = smooth_onehot
                            
                            if grid_xy_fract[1] < offset:
                                dxdy = cxcy - np.floor(cxcy - [0., 0.5])
                                grids[grid_index][batch_index][grid_xy[1]-1][grid_xy[0]][grid_anchor_index][0:4] = np.concatenate([dxdy,dwdh])
                                grids[grid_index][batch_index][grid_xy[1]-1][grid_xy[0]][grid_anchor_index][4] = 1
                                grids[grid_index][batch_index][grid_xy[1]-1][grid_xy[0]][grid_anchor_index][5:] = smooth_onehot      
                        
                        if (grid_xy<train_output_sizes[grid_index]-1).all():
                            if grid_xy_fract[0] > offset:
                                dxdy = cxcy - np.floor(cxcy + [0.5, 0.])
                                grids[grid_index][batch_index][grid_xy[1]][grid_xy[0]+1][grid_anchor_index][0:4] = np.concatenate([dxdy,dwdh])
                                grids[grid_index][batch_index][grid_xy[1]][grid_xy[0]+1][grid_anchor_index][4] = 1
                                grids[grid_index][batch_index][grid_xy[1]][grid_xy[0]+1][grid_anchor_index][5:] = smooth_onehot
                            
                            if grid_xy_fract[1] > offset:
                                dxdy = cxcy - np.floor(cxcy + [0., 0.5])
                                grids[grid_index][batch_index][grid_xy[1]+1][grid_xy[0]][grid_anchor_index][0:4] = np.concatenate([dxdy,dwdh])
                                grids[grid_index][batch_index][grid_xy[1]+1][grid_xy[0]][grid_anchor_index][4] = 1
                                grids[grid_index][batch_index][grid_xy[1]+1][grid_xy[0]][grid_anchor_index][5:] = smooth_onehot
                            
                elif int(batch_boxes[batch_index][box_index][4])!=0 & int(batch_boxes[batch_index][box_index][-1])==0:
                    anchor_index = np.argmax(iou_scores[batch_index][box_index])
                    grid_index = (anchor_index // len(anchors[0])).astype(np.int32)
                    grid_anchor_index = (anchor_index % len(anchors[0])).astype(np.int32)
                    
                    cxcy = (batch_boxes[batch_index][box_index][0:2]+batch_boxes[batch_index][box_index][2:4])/2

                    cxcy *= train_output_sizes[grid_index]
                    grid_xy = np.floor(cxcy).astype(np.int32)

                    dxdy = cxcy - grid_xy
                    dwdh = batch_boxes[batch_index][box_index][2:4]-batch_boxes[batch_index][box_index][0:2]
                    dwdh = dwdh*np.array(train_output_sizes[grid_index])
                                        
                    grids[grid_index][batch_index][grid_xy[1]][grid_xy[0]][grid_anchor_index][0:4] = np.concatenate([dxdy,dwdh])
                    grids[grid_index][batch_index][grid_xy[1]][grid_xy[0]][grid_anchor_index][4] = 1
                    grids[grid_index][batch_index][grid_xy[1]][grid_xy[0]][grid_anchor_index][5:] = smooth_onehot
                    
                    grid_xy_fract = cxcy%1.
                    
                    if (grid_xy>0).all():
                        if grid_xy_fract[0] < offset:
                            dxdy = cxcy - np.floor(cxcy - [0.5, 0.])
                            grids[grid_index][batch_index][grid_xy[1]][grid_xy[0]-1][grid_anchor_index][0:4] = np.concatenate([dxdy,dwdh])
                            grids[grid_index][batch_index][grid_xy[1]][grid_xy[0]-1][grid_anchor_index][4] = 1
                            grids[grid_index][batch_index][grid_xy[1]][grid_xy[0]-1][grid_anchor_index][5:] = smooth_onehot
                        
                        if grid_xy_fract[1] < offset:
                            dxdy = cxcy - np.floor(cxcy - [0., 0.5])
                            grids[grid_index][batch_index][grid_xy[1]-1][grid_xy[0]][grid_anchor_index][0:4] = np.concatenate([dxdy,dwdh])
                            grids[grid_index][batch_index][grid_xy[1]-1][grid_xy[0]][grid_anchor_index][4] = 1
                            grids[grid_index][batch_index][grid_xy[1]-1][grid_xy[0]][grid_anchor_index][5:] = smooth_onehot      
                    
                    if (grid_xy<train_output_sizes[grid_index]-1).all():
                        if grid_xy_fract[0] > offset:
                            dxdy = cxcy - np.floor(cxcy + [0.5, 0.])
                            grids[grid_index][batch_index][grid_xy[1]][grid_xy[0]+1][grid_anchor_index][0:4] = np.concatenate([dxdy,dwdh])
                            grids[grid_index][batch_index][grid_xy[1]][grid_xy[0]+1][grid_anchor_index][4] = 1
                            grids[grid_index][batch_index][grid_xy[1]][grid_xy[0]+1][grid_anchor_index][5:] = smooth_onehot
                            
                        if grid_xy_fract[1] > offset:
                            dxdy = cxcy - np.floor(cxcy + [0., 0.5])
                            grids[grid_index][batch_index][grid_xy[1]+1][grid_xy[0]][grid_anchor_index][0:4] = np.concatenate([dxdy,dwdh])
                            grids[grid_index][batch_index][grid_xy[1]+1][grid_xy[0]][grid_anchor_index][4] = 1
                            grids[grid_index][batch_index][grid_xy[1]+1][grid_xy[0]][grid_anchor_index][5:] = smooth_onehot
                            
    return grids
    

