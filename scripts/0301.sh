#!/bin/bash

# dataset=cifar10
# loss_type=GHMc
# train_rule=None

# for imb_factor in 500 100 10 1
# do
#     python tinyImagenet_train.py --imb_factor $imb_factor --loss_type $loss_type --train_rule $train_rule --seed 123 --beta 1.2
# done

# for train_rule in EffectiveNumber ClassBlance
# do
    
# done

# dataset=cifar100
train_rule=None
imb_factor=100
beta=1.2

for bin in 10 20 30
do
    python tinyImagenet_train.py --imb_factor $imb_factor --loss_type GHMSeesawV2 --train_rule $train_rule --seed 123 --beta $beta --bin $bin
done

# for beta in 1 1.11 1.25 1.43 1.67 2
# do
#     python cifar_train.py --dataset $dataset --imb_factor $imb_factor --loss_type GHMSeesawV2 --train_rule $train_rule --seed 123 --beta $beta
# done

# for train_rule in EffectiveNumber ClassBlance
# do
#     for imb_factor in 500 100 10 1
#     do
#         python tinyImagenet_train.py --imb_factor $imb_factor --loss_type $loss_type --train_rule $train_rule --seed 123 --beta 1.2
#     done
# done