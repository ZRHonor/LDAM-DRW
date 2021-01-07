#!/bin/bash

train_rule=None
dataset=cifar100
loss_type=SoftSeesaw
beta=0.5
# seed=

for imb_factor in 500
do
    for loss_type in SoftSeesaw Seesaw Seesaw_prior
    do
        for seed in 123
        do
            python cifar_train.py --dataset $dataset --imb_factor $imb_factor --loss_type $loss_type --train_rule $train_rule --beta $beta --seed $seed
        done
    done
done

# for loss_type in Seesaw_prior SoftSeesaw Seesaw
# do
#     for imb_factor in 100 10 1
#     do
#         python cifar_train.py --dataset $dataset --imb_factor $imb_factor --loss_type $loss_type --train_rule $train_rule
#     done
# done