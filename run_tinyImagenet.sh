
for imb_factor in 1000 500 100 10 1
do
    for loss_type in SoftSeesaw
    do
        python tinyImagenet_train.py --loss_type $loss_type --imb_factor $imb_factor --seed 123 --beta 0.5
    done
done