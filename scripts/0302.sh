#!/bin/bash
beta=1.2

for train_rule in EffectiveNumber ClassBlance
do
    for imb_factor in 500 100 10 1
    do
        python cifar_train.py --dataset cifar10 --imb_factor $imb_factor --loss_type CE --train_rule $train_rule --seed 123 --beta $beta
        python cifar_train.py --dataset cifar100 --imb_factor $imb_factor --loss_type CE --train_rule $train_rule --seed 123 --beta $beta
        python tinyImagenet_train.py --imb_factor $imb_factor --loss_type CE --train_rule $train_rule --seed 123 --beta $beta
    done
done