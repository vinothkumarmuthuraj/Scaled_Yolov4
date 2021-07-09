# Scaled_Yolov4
A Tensorflow2.x implementation of Scaled-YOLOv4 as described in Scaled-YOLOv4: Scaling Cross Stage Partial Network

# Demo
# Yolov4_tiny
python train.py --model_type tiny --train_annot_directory ./train/annot --train_image_directory ./train/image --valid_annot_directory ./valid/annot --valid_image_directory ./valid/image --class_names ./class.names --train_batch_size 5 --valid_batch_size 3 --drop_block True --fixed_scale_bool True --fixed_scale 608 --data_augment None lrtp random_affine --box_regression_loss ciou --classification_loss focal --optimizer Adam --total_epochs 25 --warmup_epochs 5 --checkpoint_dir ./weight --epochvsloss_plot_path ./epochvsloss_plot_path

python detect.py --model_type tiny --image_directory ./image_directory --img_save_path ./img_save_path --checkpoint_dir ./weight --class_names ./class.names --optimizer Adam --fixed_scale_bool True --fixed_scale 608 --drop_block True

python detect_video.py --model_type tiny --video_directory ./video_directory/project.mp4 --video_ouptut_directory ./output_video --checkpoint_dir ./weight --class_names ./class.names --fixed_scale_bool True --fixed_scale 608 --drop_block True --output_video_name project

# Detection result of detecting Furniture, Doors, Garments, Cables in view of Vaccum cleaner

![vac216](https://user-images.githubusercontent.com/48114165/125116888-05047480-e10b-11eb-9549-936119563c35.jpg)
![vac232](https://user-images.githubusercontent.com/48114165/125117030-37ae6d00-e10b-11eb-89bf-adb8d25232b3.jpg)
![vac377](https://user-images.githubusercontent.com/48114165/125062652-432e7380-e0cc-11eb-9532-526dc5330de5.jpg)
![vac366](https://user-images.githubusercontent.com/48114165/125062737-59d4ca80-e0cc-11eb-9827-8dd31d77c5b1.jpg)
![vac11](https://user-images.githubusercontent.com/48114165/125062859-783ac600-e0cc-11eb-8aad-801e876989c5.jpg)


# References
Yolov4 - https://arxiv.org/pdf/2004.10934.pdf                                                                                                                                             
Scaled Yolov4 - https://arxiv.org/pdf/2011.08036.pdf                                                                                                                                     
https://github.com/hunglc007/tensorflow-yolov4-tflite
