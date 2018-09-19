#!/usr/bin/env bash

source ../hw1/bin/activate

for b in 5000
do
    for r in 0.0001 0.001 0.01 0.1
    do
        tailtag="b$b""_r$r"
        echo "b = $b, r = $r, tail_tag = $tailtag"
        cmd="python train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 3 -l 2 \
            -s 64 -b $b -lr $r -rtg --exp_name hc_$tailtag"
        echo $cmd
        eval $cmd
    done
done

# PLOT
#python plot.py data/hc_b5000_r0.1_InvertedPendulum-v2_19-09-2018_00-02-46 \
#    data/hc_b5000_r0.01_InvertedPendulum-v2_19-09-2018_00-15-13 \
#    data/hc_b5000_r0.001_InvertedPendulum-v2_19-09-2018_00-11-37 \
#    data/hc_b5000_r0.0001_InvertedPendulum-v2_19-09-2018_00-07-54

for b in 50 500 5000 50000
do
    for r in 0.01
    do
        tailtag="b$b""_r$r"
        echo "b = $b, r = $r, tail_tag = $tailtag"
        cmd="python train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 3 -l 2 \
            -s 64 -b $b -lr $r -rtg --exp_name hc_$tailtag"
        echo $cmd
        eval $cmd
    done
done

# PLOT
#python plot.py data/hc_b50_r0.01_InvertedPendulum-v2_19-09-2018_00-29-38 \
#    data/hc_b500_r0.01_InvertedPendulum-v2_19-09-2018_00-30-14 \
#    data/hc_b5000_r0.01_InvertedPendulum-v2_19-09-2018_00-30-54 \
#    data/hc_b50000_r0.01_InvertedPendulum-v2_19-09-2018_00-34-47

for b in 800 1000 2000 5000
do
    for r in 0.01
    do
        tailtag="b$b""_r$r"
        echo "b = $b, r = $r, tail_tag = $tailtag"
        cmd="python train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 3 -l 2 \
            -s 64 -b $b -lr $r -rtg --exp_name hc_$tailtag"
        echo $cmd
        eval $cmd
    done
done

#python plot.py data/hc_b800_r0.01_InvertedPendulum-v2_19-09-2018_01-15-27 \
#    data/hc_b1000_r0.01_InvertedPendulum-v2_19-09-2018_01-16-19 \
#    data/hc_b2000_r0.01_InvertedPendulum-v2_19-09-2018_01-17-31 \
#    data/hc_b5000_r0.01_InvertedPendulum-v2_19-09-2018_01-19-20

for b in 1000
do
    for r in 0.005 0.01 0.02 0.04
    do
        tailtag="b$b""_r$r"
        echo "b = $b, r = $r, tail_tag = $tailtag"
        cmd="python train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 3 -l 2 \
            -s 64 -b $b -lr $r -rtg --exp_name hc_$tailtag"
        echo $cmd
        eval $cmd
    done
done

#python plot.py data/hc_b1000_r0.005_InvertedPendulum-v2_19-09-2018_01-45-54   \
#    data/hc_b1000_r0.01_InvertedPendulum-v2_19-09-2018_01-47-04 \
#    data/hc_b1000_r0.02_InvertedPendulum-v2_19-09-2018_01-48-25 \
#    data/hc_b1000_r0.04_InvertedPendulum-v2_19-09-2018_01-49-44 --value MaxReturn


for b in 1000
do
    for r in 0.02 0.03 0.04
    do
        tailtag="b$b""_r$r"
        echo "b = $b, r = $r, tail_tag = $tailtag"
        cmd="python train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 3 -l 2 \
            -s 64 -b $b -lr $r -rtg --exp_name hc_$tailtag"
        echo $cmd
        eval $cmd
    done
done

#python plot.py data/hc_b1000_r0.02_InvertedPendulum-v2_19-09-2018_01-58-34 \
#        data/hc_b1000_r0.03_InvertedPendulum-v2_19-09-2018_01-59-53 \
#        data/hc_b1000_r0.04_InvertedPendulum-v2_19-09-2018_02-01-03 --value MaxReturn

for b in 800 900 1000
do
    for r in 0.03
    do
        tailtag="b$b""_r$r"
        echo "b = $b, r = $r, tail_tag = $tailtag"
        cmd="python train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 3 -l 2 \
            -s 64 -b $b -lr $r -rtg --exp_name hc_$tailtag"
        echo $cmd
        eval $cmd
    done
done

#python plot.py data/hc_b800_r0.03_InvertedPendulum-v2_19-09-2018_02-05-46 \
#    data/hc_b900_r0.03_InvertedPendulum-v2_19-09-2018_02-06-34 \
#    data/hc_b1000_r0.03_InvertedPendulum-v2_19-09-2018_02-07-30
#
#python plot.py data/hc_b900_r0.03_InvertedPendulum-v2_19-09-2018_02-06-34



