#!/usr/bin/env bash

source ../hw1/bin/activate


python train_pg_f18.py LunarLanderContinuous-v2 -ep 1000 --discount 0.99 -n \
    100 -e 3 -l 2 -s 64 -b 40000 -lr 0.005 \
    -rtg --nn_baseline --exp_name ll_b40000_r0.005

python plot.py data/ll_b40000_r0.005_LunarLanderContinuous-v2_18-09-2018_16-51-30