#!/bin/bash

for imb_factor in 500
do
    for loss_type in CE
    do
        CUDA_VISIBLE_DEVICES='4,5,6,7' python Imagenet_train.py --imb_factor $imb_factor --loss_type $loss_type
    done
done
