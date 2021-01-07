#!/bin/bash

# CIFAR100
# train_rule=None
# dataset=cifar100

# for imb_factor in 500 100 10 1
# do
#     for loss_type in SoftSeesaw
#     do
#         python cifar_train.py --dataset $dataset --imb_factor $imb_factor --loss_type $loss_type --train_rule $train_rule --seed 123
#     done
# done

# CIFAR10
# train_rule=None
# dataset=cifar10

# for imb_factor in 1 10 100 500
# do
#     for loss_type in CE Seesaw SoftSeesaw EQL
#     do
#         python cifar_train.py --dataset $dataset --imb_factor $imb_factor --loss_type $loss_type --train_rule $train_rule --seed 123
#     done
#     python cifar_train.py --dataset $dataset --imb_factor $imb_factor --loss_type CE --train_rule ClassBlance --seed 123
# done

# TinyImagenet


for imb_factor in 1000 500 100 10 1
do
    for loss_type in Seesaw
    do
        python tinyImagenet_train.py --loss_type $loss_type --imb_factor $imb_factor --seed 123
    done
done

