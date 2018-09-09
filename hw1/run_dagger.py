#!/usr/bin/env python

"""
Code to load a expert policy and learn in a dagger style.
Example usage:
    python run_expert.py experts/Humanoid-v2.pkl Humanoid-v2 --render \
            --num_rollouts 20

Author: Naijia Fan
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import gym
import load_policy
import load_expert


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('expert_data_file', type=str)
    parser.add_argument('cloning_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument("--dagger_iterations", type=int)
    args = parser.parse_args()

    policy_fn = load_policy.load_policy(args.expert_policy_file)
    x, y = load_expert.load_expert_data(args.expert_data_file)

    mean_rewards = []
    std_rewards = []

    for idx in range(args.dagger_iterations):
        print('dagger round %i/%i' % (idx, args.dagger_iterations))
        x_train, x_test, y_train, y_test = load_expert.split_expert_data(x, y)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            env = gym.make(args.envname)
            max_steps = args.max_timesteps or env.spec.timestep_limit

            returns = []
            add_observations = []
            add_actions = []

            model = keras.Sequential([
                keras.layers.Dense(128, activation=tf.nn.relu,
                                   input_shape=(x.shape[1],)),
                keras.layers.Dense(128, activation=tf.nn.relu),
                keras.layers.Dense(128, activation=tf.nn.relu),
                keras.layers.Dense(y.shape[1])
            ])
            model.compile(optimizer='adam', loss="mse", metrics=['mse'])
            model.fit(x_train, y_train, epochs=10, verbose=0)

            for i in range(args.num_rollouts):
                # print('iter', i)
                obs = env.reset()
                done = False
                totalr = 0.
                steps = 0
                while not done:
                    expert_action = policy_fn(obs[None, :])
                    action = model.predict(obs.reshape(1, -1))
                    add_observations.append(obs)
                    add_actions.append(expert_action)
                    obs, r, done, _ = env.step(action)
                    totalr += r
                    steps += 1
                    if args.render:
                        env.render()
                    # if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                    if steps >= max_steps:
                        break
                returns.append(totalr)

            # print('returns', returns)
            # print('mean return', np.mean(returns))
            # print('std of return', np.std(returns))
            mean_rewards.append(np.mean(returns))
            std_rewards.append(np.std(returns))

        add_observations = np.array(add_observations)
        add_actions = np.array(add_actions)

        # every time dagger, get more expert data
        add_observations = add_observations.reshape(add_observations.shape[0], -1)
        add_actions = add_actions.reshape(add_actions.shape[0], -1)
        x = np.concatenate([x, add_observations])
        y = np.concatenate([y, add_actions])

    print('mean returns', mean_rewards)
    print('std returns', std_rewards)


if __name__ == '__main__':
    main()
