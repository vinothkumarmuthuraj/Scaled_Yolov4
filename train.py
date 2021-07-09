#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import cv2
import copy
import math
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# In[ ]:


from generator.get_generator import get_generator

from model.yolov4 import Yolov4

from model.yolov4_tiny import Yolov4_tiny

from model.losses import yolov3_loss


# In[ ]:


def parse_args(args):
    parser = argparse.ArgumentParser(description='Simple training script for using ScaledYOLOv4.')
    parser.add_argument('--dataset_type', default='voc')
    parser.add_argument('--model_type', default='tiny',help="choices=['tiny','p5','p6','p7']")
    parser.add_argument('--train_annot_directory', default='./train_annot')
    parser.add_argument('--train_image_directory', default='./train_image')
    parser.add_argument('--valid_annot_directory', default='./valid_annot')
    parser.add_argument('--valid_image_directory', default='./valid_image')
    parser.add_argument('--class_names', default='class_names.names',help="voc.names")
    parser.add_argument('--train_batch_size', default=5, type=int)
    parser.add_argument('--valid_batch_size', default=3, type=int)
    parser.add_argument('--drop_block', default='False', help="choices=[True,False]")
    parser.add_argument('--fixed_scale_bool', default='False', type=bool,help="choices=[True,False]")
    parser.add_argument('--fixed_scale', default=608, type=int)
    parser.add_argument('--multi_scale', default=512, nargs='+', type=int, help="Input data shapes for training, use 320+32*i(i>=0),eg: 416 512 640")
    parser.add_argument('--data_augment', default='lrtp',nargs='+', help="choices=['None','lrtp','random_affine','random_color_distort']")
    parser.add_argument('--anchor_match_iou_thr', default=0.2, type=float)
    parser.add_argument('--offset', default=0.5, type=float)
    parser.add_argument('--scales_x_y', default=[2., 2., 2., 2., 2.])
    parser.add_argument('--focal_alpha', default= 0.25, type=float)
    parser.add_argument('--focal_gamma', default=2.0, type=float)
    parser.add_argument('--box_regression_loss', default='ciou',help="choices=['giou','diou','ciou']")
    parser.add_argument('--classification_loss', default='bce', help="choices=[bce','focal']")
    parser.add_argument('--optimizer', default='Adam', help="choices=[Adam,sgd]")
    parser.add_argument('--total_epochs', default=25, type=int)
    parser.add_argument('--warmup_epochs', default=2, type=int)
    parser.add_argument('--train_lr_init', default=0.001, type=float)
    parser.add_argument('--train_lr_end', default=0.00001, type=float)
    parser.add_argument('--reg_losss_weight', default=0.05, type=float)
    parser.add_argument('--obj_losss_weight', default=1., type=float)
    parser.add_argument('--cls_losss_weight', default=0.5, type=float)
    parser.add_argument('--save_step', default=2, type=int)
    parser.add_argument('--save_method', default='.ckpt',help="choices=[.ckpt,.h5]")
    parser.add_argument('--checkpoint_dir', default='./checkpoint')
    parser.add_argument('--epochvsloss_plot_path', default='./epochvsloss_plot_path')
    
    
    return parser.parse_args(args)


# In[1]:


def grad(model,image_data,y_true,loss_fun,training=True):
    with tf.GradientTape() as tape:
        pred_result = model(image_data, training)
        reg_loss = conf_loss = cls_loss = 0
#         print('y_true_length',len(y_true))
        # optimizing process
        for i in range(len(y_true)):
#             print('y_true',y_true[i].shape)
#             print('pred_result',pred_result[i].shape)
            loss_items = loss_fun[i](y_true[i],pred_result[i])
            reg_loss += loss_items[0]
            conf_loss += loss_items[1]
            cls_loss += loss_items[2]
            
        total_loss = reg_loss + conf_loss + cls_loss
    return total_loss, reg_loss,conf_loss,cls_loss, tape.gradient(total_loss, model.trainable_variables)


# In[ ]:


def main(args):
    
    train_lr_init = args.train_lr_init
    train_lr_end = args.train_lr_end
    
    Train_annot_folder=os.listdir(args.train_annot_directory)
    Total_train=len(Train_annot_folder)
    steps_per_epoch_train=Total_train//args.train_batch_size
                                       
    valid_annot_folder = os.listdir(args.valid_annot_directory)
    Total_valid = len(valid_annot_folder)
    steps_per_epoch_valid = Total_valid//args.valid_batch_size
                                         
    warmup_steps = args.warmup_epochs * steps_per_epoch_train
    total_steps = args.total_epochs * steps_per_epoch_train
                                         
    num_model_outputs = {"tiny":2, "p5":3,"p6":4,"p7":5}
    loss_fun = [yolov3_loss(args,grid_index) for grid_index in range(num_model_outputs[args.model_type])]
    
    train_generator, valid_generator = get_generator(args)
    
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
                                         
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
        
    #checkpoint
    root = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer,model=model)
    
    if args.save_method == '.ckpt':
        manager = tf.train.CheckpointManager(root, args.checkpoint_dir, max_to_keep=3)
        root.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print("Restored from ckpt {}".format(manager.latest_checkpoint))
        else:
            print("Initializing from scratch ckpt.")
    
    elif args.save_method == '.h5':
        file_paths = os.listdir(args.checkpoint_dir)

        if len(file_paths) == 0:
            print('Initializing from scratch h5.')
        else:
            for File in file_paths:
                if File.endswith(".h5"):
                    model.load_weights(args.checkpoint_dir + '\yolo.h5',by_name=True, skip_mismatch=True)
                    print('Restored from .h5')

        
    train_loss_history = []
    val_loss_history = []
        
    # training
    for epoch in range(args.total_epochs):
        epoch_loss = []
        epoch_reg_loss = []
        epoch_conf_loss = []
        epoch_cls_loss=[]
        
        epoch_val_loss=[]
        epoch_val_reg_loss = []
        epoch_val_conf_loss = []
        epoch_val_cls_loss = []
        print('Epoch {} :'.format(epoch))
        
        for batch_idx in range(steps_per_epoch_train): 
            train_img,train_y_true,train_batch_boxes =  next(train_generator)
            total_loss, reg_loss,conf_loss,cls_loss, grads = grad(model,train_img, train_y_true,loss_fun)
            
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss.append(total_loss)
            epoch_reg_loss.append(reg_loss)
            epoch_conf_loss.append(conf_loss)
            epoch_cls_loss.append(cls_loss)                           
            if global_steps < warmup_steps:
                lr = global_steps / warmup_steps * train_lr_init
                                         
            else:
                lr = train_lr_end + 0.5 * (train_lr_init - train_lr_end) * (
                    (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi)))
                                         
            optimizer.lr.assign(lr.numpy())
            global_steps.assign_add(1)
                                         
            print('-',end='')
        print(' | ',end='')
                                         
        for val_batch_idx in range(steps_per_epoch_valid): 
            val_img, val_y_true,val_batch_boxes =  next(valid_generator)
            val_total_loss, val_reg_loss,val_conf_loss,val_cls_loss, val_grads = grad(model,val_img,val_y_true,loss_fun)
            epoch_val_loss.append(val_total_loss)
            epoch_val_reg_loss.append(val_reg_loss)
            epoch_val_conf_loss.append(val_conf_loss)
            epoch_val_cls_loss.append(val_cls_loss)

            print('-',end='')
        print(' | ',end='')
                                         
        train_loss_avg = np.mean(np.array(epoch_loss))
        reg_loss_avg = np.mean(np.array(epoch_reg_loss))
        conf_loss_avg = np.mean(np.array(epoch_conf_loss))
        cls_loss_avg = np.mean(np.array(epoch_cls_loss))
        
        val_loss_avg=np.mean(np.array(epoch_val_loss))
        val_reg_loss_avg = np.mean(np.array(epoch_val_reg_loss))
        val_conf_loss_avg = np.mean(np.array(epoch_val_conf_loss))
        val_cls_loss_avg = np.mean(np.array(epoch_val_cls_loss))      
        
        train_loss_history.append(train_loss_avg)
        val_loss_history.append(val_loss_avg)
                                         
        print(' loss = {:.4f}, reg_loss_avg = {:.4f}, conf_loss_avg={:.4f}, class_loss_avg={:.4f}, val_loss={:.4f}, val_reg_loss_avg = {:.4f}, val_conf_loss_avg = {:.4f},val_cls_loss_avg = {:.4f},lr={:.4f})'.format(
            train_loss_avg, reg_loss_avg, conf_loss_avg, cls_loss_avg, val_loss_avg,
            val_reg_loss_avg, val_conf_loss_avg, val_cls_loss_avg, optimizer.lr.numpy()))
                                         
        root.step.assign_add(1)
        if args.save_method == '.ckpt':
            if int(root.step) % args.save_step == 0:
                save_path = manager.save()
                print("Saved checkpoint for step {}: {}".format(int(root.step), save_path))
                
        elif args.save_method == '.h5':
            if int(root.step) % args.save_step == 0:
                model.save_weights(args.checkpoint_dir + '\yolo.h5')
                print('model_saved')
    
    fig = plt.figure()
    plt.plot(train_loss_history)
    plt.plot(val_loss_history)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and valid loss')
    plt.legend(['Train_loss','validation_loss'])
    fig.savefig(args.epochvsloss_plot_path,facecolor='y',bbox_inches='tight',pad_inches=0.3,transparent=True)
    
    return [train_loss_history,val_loss_history]


# In[ ]:


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)

