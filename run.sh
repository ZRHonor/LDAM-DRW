#!/bin/bash


# dataset=cifar100
# loss_type=CE
# train_rule=None

# for imb_factor in 1 0.1 0.01
# do
#     python cifar_train.py --dataset $dataset --imb_factor $imb_factor --loss_type $loss_type --train_rule $train_rule
# done

# dataset=cifar100
# loss_type=LDAM
# train_rule=None

# for imb_factor in 1 0.1 0.01
# do
#     python cifar_train.py --dataset $dataset --imb_factor $imb_factor --loss_type $loss_type --train_rule $train_rule
# done


# dataset=cifar100
# loss_type=CE
# train_rule=EffectiveNumber

# for imb_factor in 1 0.1 0.01
# do
#     python cifar_train.py --dataset $dataset --imb_factor $imb_factor --loss_type $loss_type --train_rule $train_rule
# done


# dataset=cifar100
# loss_type=CE
# train_rule=ClassBlance

# for imb_factor in 0.1 0.01
# do
#     python cifar_train.py --dataset $dataset --imb_factor $imb_factor --loss_type $loss_type --train_rule $train_rule
# done

# dataset=cifar100
# loss_type=Focal
# train_rule=None

# for imb_factor in 1 0.1 0.01
# do
#     python cifar_train.py --dataset $dataset --imb_factor $imb_factor --loss_type $loss_type --train_rule $train_rule
# done

dataset=cifar100
loss_type=Seesaw_prior
train_rule=None

for imb_factor in 1 0.1 0.01
do
    python cifar_train.py --dataset $dataset --imb_factor $imb_factor --loss_type $loss_type --train_rule $train_rule
done


dataset=cifar100
loss_type=Seesaw
train_rule=None

for imb_factor in 1 0.1 0.01
do
    python cifar_train.py --dataset $dataset --imb_factor $imb_factor --loss_type $loss_type --train_rule $train_rule
done

