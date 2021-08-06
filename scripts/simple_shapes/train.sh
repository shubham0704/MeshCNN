#!/usr/bin/env bash

## run the training
python demo_train.py \
--dataroot datasets/basic_shapes \
--name basic_shapes \
--arch meshunet \
--dataset_mode segmentation \
# --ninput_edges 16 \ # if we supply this more that what we have in mesh they will do 0 padding
# --ncf 32 64 128 256 \
# --pool_res 1800 1350 600 \
--resblocks 3 \
--batch_size 1 \
--lr 0.001 \
# --flip_edges 0.2 \
# --slide_verts 0.2 \
# --num_aug 20 \
