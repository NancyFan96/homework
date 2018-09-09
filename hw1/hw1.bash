#!/usr/bin/env bash
#set -eux
source ./bin/activate
declare -a arr=(Ant-v2 HalfCheetah-v2 Hopper-v2 Humanoid-v2 Reacher-v2 Walker2d-v2)

# algorithm 1. expert policy
for e in ${arr[@]}
do
    python run_expert.py experts/$e.pkl $e --render --num_rollouts 20
done

# algorithm 2. behavioral cloning
# (1) learn cloning
for e in  ${arr[@]}
do
    python behavioral_cloning.py expert_data/$e.pkl $e
done

# (2) run cloning
for e in ${arr[@]}
do
    python run_cloning.py cloning_policy/$e.h5 $e --render --num_rollouts 20
done

# (3) test related paramter
for epochs in 1 3 5 7 9 11 13
do
    python behavioral_cloning.py expert_data/Ant-v2.pkl Ant-v2-$epochs --epochs $epochs
    python run_cloning.py cloning_policy/Ant-v2-$epochs.h5 Ant-v2 --render --num_rollouts 20
done

# algorithm 3. dagger
python run_dagger.py experts/Hopper-v2.pkl expert_data/Hopper-v2.pkl cloning_policy/Hopper-v2.h5 Hopper-v2 --num_rollouts 20 --dagger_iterations 10
