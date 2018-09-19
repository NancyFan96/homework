#!/usr/bin/env bash

source ../hw1/bin/activate

# b: 10000, 30000, 50000
# r: 0.005, 0.01, 0.02


for b in 10000 30000 50000
do
    for r in 0.005 0.01 0.02
    do
        tailtag="b$b""_r$r"
        echo "b = $b, r = $r, tail_tag = $tailtag"
        cmd="python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 \
            -s 32 -b $b -lr $r -rtg --nn_baseline --exp_name hc_$tailtag"
        echo $cmd
        eval $cmd
    done
done

# PLOT & COMPARE
#python plot.py data/hc_b10000_r0.005_HalfCheetah-v2_18-09-2018_19-43-39 \
#    data/hc_b10000_r0.01_HalfCheetah-v2_18-09-2018_19-51-50 \
#    data/hc_b10000_r0.02_HalfCheetah-v2_18-09-2018_20-00-22 \
#    data/hc_b30000_r0.005_HalfCheetah-v2_18-09-2018_20-08-09 \
#    data/hc_b30000_r0.01_HalfCheetah-v2_18-09-2018_20-31-07 \
#    data/hc_b30000_r0.02_HalfCheetah-v2_18-09-2018_20-54-17 \
#    data/hc_b50000_r0.005_HalfCheetah-v2_18-09-2018_21-18-04 \
#    data/hc_b50000_r0.01_HalfCheetah-v2_18-09-2018_21-58-46 \
#    data/hc_b50000_r0.02_HalfCheetah-v2_18-09-2018_22-39-10
#
#python plot.py data/hc_b10000_r0.005_HalfCheetah-v2_18-09-2018_19-43-39 \
#    data/hc_b10000_r0.01_HalfCheetah-v2_18-09-2018_19-51-50 \
#    data/hc_b10000_r0.02_HalfCheetah-v2_18-09-2018_20-00-22 \
#    data/hc_b30000_r0.005_HalfCheetah-v2_18-09-2018_20-08-09 \
#    data/hc_b30000_r0.01_HalfCheetah-v2_18-09-2018_20-31-07 \
#    data/hc_b30000_r0.02_HalfCheetah-v2_18-09-2018_20-54-17 \
#    data/hc_b50000_r0.005_HalfCheetah-v2_18-09-2018_21-18-04 \
#    data/hc_b50000_r0.01_HalfCheetah-v2_18-09-2018_21-58-46 \
#    data/hc_b50000_r0.02_HalfCheetah-v2_18-09-2018_22-39-10 --value MaxReturn


bb=50000
rr=0.02
baserun="python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b $bb -lr $rr"
tailtag="b$bb""_r$rr"

echo $baserun --exp_name hc_$tailtag
echo $baserun -rtg --exp_name hc_rtg_$tailtag
echo $baserun --nn_baseline --exp_name hc_nnb_$tailtag
echo $baserun -rtg --nn_baseline --exp_name hc_rtg_nnb_$tailtag

eval $baserun --exp_name hc_$tailtag
eval $baserun -rtg --exp_name hc_rtg_$tailtag
eval $baserun --nn_baseline --exp_name hc_nnb_$tailtag
eval $baserun -rtg --nn_baseline --exp_name hc_rtg_nnb_$tailtag


#python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 \
#    -l 2 -s 32 -b <b*> -lr <r*> --exp_name hc_b<b*>_r<r*>
#python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 \
#    -l 2 -s 32 -b <b*> -lr <r*> -rtg --exp_name hc_b<b*>_r<r*>
#python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 \
#    -l 2 -s 32 -b <b*> -lr <r*> --nn_baseline --exp_name hc_b<b*>_r<r*>
#python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 \
#    -l 2 -s 32 -b <b*> -lr <r*> -rtg --nn_baseline --exp_name hc_b<b*>_r<r*>

#testb="ls "
#eval $testb -a