#!/usr/bin/env python

"""
Code to load a behaviour cloning policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_cloning.py experts/Humanoid-v2.pkl Humanoid-v2 --render \
            --num_rollouts 20
Author: Naijia Fan
"""

import os
import pickle
import tensorflow as tf
from tensorflow import keras
import numpy as np
import gym
import tf_util


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('cloning_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('loading and building behavioral cloning')
    bc_model = keras.models.load_model(args.cloning_policy_file)
    print('loaded and built')


    import gym
    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit

    returns = []
    observations = []
    actions = []
    for i in range(args.num_rollouts):
        # print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = bc_model.predict(obs.reshape(1, -1))
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if args.render:
                env.render()
            # if steps % 100 == 0:print("%i/%i" % (steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

    behavior_cloning_data = {'observations': np.array(observations),
                             'actions': np.array(actions)}

    with open(os.path.join('expert_data/bc', args.envname + '.pkl'), 'wb') as f:
        pickle.dump(behavior_cloning_data, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
