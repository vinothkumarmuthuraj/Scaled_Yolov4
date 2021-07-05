#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from generator.generator_builder import VocGenerator


# In[ ]:


def get_generator(args):
    if args.dataset_type == 'voc':
        train_dataset = VocGenerator(args,mode=0)
        valid_dataset = VocGenerator(args,mode=1)
    else:
        raise ValueError("{} is invalid!".format(args['dataset_type']))
        
    return train_dataset,valid_dataset

