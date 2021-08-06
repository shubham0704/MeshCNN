#!/usr/bin/env bash

## run the training
python demo_train.py \
--dataroot datasets/synthetic_tower_data \
--name tower_seg \
--arch meshunet \
--dataset_mode segmentation \
--ncf 32 64 128 256 \
--ninput_edges 44430 \
--pool_res 20000 10000 5000 \
--resblocks 3 \
--batch_size 2 \
--lr 0.001 \
--num_aug 20 \
--slide_verts 0.2 \