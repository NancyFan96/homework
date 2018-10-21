#!/usr/bin/env bash

source ../294-112/bin/activate


#python train_ac_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.95 -n 100 -e 3 -l 2 -s 64 -b 5000 -lr 0.01 \
#    --exp_name ip_1_100 -ntu 1 -ngsptu 100
#
#python train_ac_f18.py HalfCheetah-v2 -ep 150 --discount 0.90 -n 100 -e 3 -l 2 -s 32 -b 30000 -lr 0.02 \
#    --exp_name hc_1_100 -ntu 1 -ngsptu 100
#
#python train_ac_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.95 -n 100 -e 3 -l 2 -s 64 -b 5000 -lr 0.01 \
#    --exp_name ip_10_10 -ntu 10 -ngsptu 10

python train_ac_f18.py HalfCheetah-v2 -ep 150 --discount 0.90 -n 100 -e 3 -l 2 -s 32 -b 30000 -lr 0.02 \
    --exp_name hc_10_10 -ntu 10 -ngsptu 10