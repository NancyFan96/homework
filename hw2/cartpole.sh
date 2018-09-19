#!/usr/bin/env bash
#set -eux
source ../hw1/bin/activate

python train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 -dna --exp_name sb_no_rtg_dna
python train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 -rtg -dna --exp_name sb_rtg_dna
python train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 -rtg --exp_name sb_rtg_na
python train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 3 -dna --exp_name lb_no_rtg_dna
python train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 3 -rtg -dna --exp_name lb_rtg_dna
python train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 3 -rtg --exp_name lb_rtg_na

# PLOT
#python plot.py data/lb_no_rtg_dna_CartPole-v0_18-09-2018_15-13-53 data/lb_rtg_dna_CartPole-v0_18-09-2018_15-16-40 data/lb_rtg_na_CartPole-v0_18-09-2018_15-19-29
#python plot.py data/sb_no_rtg_dna_CartPole-v0_18-09-2018_15-11-54 data/sb_rtg_dna_CartPole-v0_18-09-2018_15-12-35 data/sb_rtg_na_CartPole-v0_18-09-2018_15-13-17
