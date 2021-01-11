#!/bin/bash

for imb_factor in 100
do
    for beta in 0.85 0.9 0.95
    do
        python tinyImagenet_train.py --loss_type GHMSeesawV2 --imb_factor $imb_factor --seed 123 --beta $beta
    done
    # python tinyImagenet_train.py --loss_type $loss_type --imb_factor $imb_factor --seed 123
done

for imb_factor in 500 200 100
do
    for beta in 1.1 1.3 1.4 1.5
    do
        python tinyImagenet_train.py --loss_type GHMSeesawV2 --imb_factor $imb_factor --seed 123 --beta $beta
    done
    # python tinyImagenet_train.py --loss_type $loss_type --imb_factor $imb_factor --seed 123
done