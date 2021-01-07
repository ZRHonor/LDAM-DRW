#!/bin/bash

train_rule=None
dataset=cifar100

for imb_factor in 1 10 100
do
    for loss_type in SoftSeesaw
    do
        python cifar_train.py --dataset $dataset --imb_factor $imb_factor --loss_type $loss_type --train_rule $train_rule --seed 123
    done
done

# for loss_type in Seesaw_prior SoftSeesaw Seesaw
# do
#     for imb_factor in 100 10 1
#     do
#         python cifar_train.py --dataset $dataset --imb_factor $imb_factor --loss_type $loss_type --train_rule $train_rule
#     done
# done