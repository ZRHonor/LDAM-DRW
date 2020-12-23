#!/bin/bash

train_rule=None
dataset=cifar10
loss_type=SoftSeesaw

for imb_factor in 1000
do
    for beta in 0.5
    do
        python cifar_train.py --dataset $dataset --imb_factor $imb_factor --loss_type $loss_type --train_rule $train_rule --beta $beta
    done
done

# for loss_type in Seesaw_prior SoftSeesaw Seesaw
# do
#     for imb_factor in 100 10 1
#     do
#         python cifar_train.py --dataset $dataset --imb_factor $imb_factor --loss_type $loss_type --train_rule $train_rule
#     done
# done