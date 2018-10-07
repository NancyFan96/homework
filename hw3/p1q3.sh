#!/usr/bin/env bash

# fix seed as 1825
python run_dqn_atari.py -m 0.1 --fixed_seed
python run_dqn_atari.py -m 10 --fixed_seed
python run_dqn_atari.py -m 100 --fixed_seed
