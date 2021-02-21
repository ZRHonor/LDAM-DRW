#!/bin/bash

# CIFAR100
train_rule=None
dataset=cifar100

for imb_factor in 1 10 100 200 500
do
    for loss_type in GHMSeesawV2 FocalLoss GHMcLoss
    do
        python cifar_train.py --dataset $dataset --imb_factor $imb_factor --loss_type $loss_type --train_rule $train_rule --seed 123 --beta 1.3
    done
    python cifar_train.py --dataset $dataset --imb_factor $imb_factor --loss_type CE --train_rule ClassBlance --seed 123
done

# CIFAR10
train_rule=None
dataset=cifar10

for imb_factor in 1 10 100 200 500
do
    for loss_type in GHMSeesawV2 FocalLoss GHMcLoss
    do
        python cifar_train.py --dataset $dataset --imb_factor $imb_factor --loss_type $loss_type --train_rule $train_rule --seed 123 --beta 1.3
    done
done

TinyImagenet


# for imb_factor in 1 10 100 200 500 1000
# do
#     for loss_type in GHMSeesawV2
#     do
#         python tinyImagenet_train.py --loss_type $loss_type --imb_factor $imb_factor --seed 123
#     done
#     # python tinyImagenet_train.py --loss_type $loss_type --imb_factor $imb_factor --seed 123
# done

