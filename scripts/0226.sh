#!/bin/bash

dataset=cifar10
loss_type=CE

for train_rule in EffectiveNumber ClassBlance
do
    for imb_factor in 500 100 10 1
    do
        python cifar_train.py --dataset $dataset --imb_factor $imb_factor --loss_type $loss_type --train_rule $train_rule --seed 123 --beta 1.2
    done
done

dataset=cifar100
# train_rule=None

for train_rule in EffectiveNumber ClassBlance
do
    for imb_factor in 500 100 10 1
    do
        python cifar_train.py --dataset $dataset --imb_factor $imb_factor --loss_type $loss_type --train_rule $train_rule --seed 123 --beta 1.2
    done
done

for train_rule in EffectiveNumber ClassBlance
do
    for imb_factor in 500 100 10 1
    do
        python tinyImagenet_train.py --imb_factor $imb_factor --loss_type $loss_type --train_rule $train_rule --seed 123 --beta 1.2
    done
done