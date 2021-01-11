#!/bin/bash

# CIFAR100
# train_rule=None
# dataset=cifar100

# for imb_factor in 200
# do
#     for loss_type in CE Seesaw EQL
#     do
#         python cifar_train.py --dataset $dataset --imb_factor $imb_factor --loss_type $loss_type --train_rule $train_rule --seed 123
#     done
#     python cifar_train.py --dataset $dataset --imb_factor $imb_factor --loss_type CE --train_rule ClassBlance --seed 123
# done

# CIFAR10
# train_rule=None
# dataset=cifar10

# for imb_factor in 1 10 100 200 500
# do
#     for loss_type in CE Seesaw GHMSeesawV2 EQL
#     do
#         python cifar_train.py --dataset $dataset --imb_factor $imb_factor --loss_type $loss_type --train_rule $train_rule --seed 123
#     done
#     python cifar_train.py --dataset $dataset --imb_factor $imb_factor --loss_type CE --train_rule ClassBlance --seed 123
# done

# TinyImagenet


for imb_factor in 1 10 100 200 500 1000
do
    for loss_type in GHMSeesawV2
    do
        python tinyImagenet_train.py --loss_type $loss_type --imb_factor $imb_factor --seed 123
    done
    # python tinyImagenet_train.py --loss_type $loss_type --imb_factor $imb_factor --seed 123
done

