#!/usr/bin/env python

"""
Code to clone an expert policy and generate behavioral cloning policy for furture use.
Example usage:
    python behavioral_cloning.py expert_data/Humanoid-v2.pkl Humanoid-v2

Author: Naijai Fan
"""

import os
import tensorflow as tf
from tensorflow import keras
import load_expert


def _clone_behavior(x, y, epochs, envname):
    x_train, x_test, y_train, y_test = load_expert.split_expert_data(x, y)
    model = keras.Sequential([
        keras.layers.Dense(128, activation=tf.nn.relu,
                           input_shape=(x.shape[1], )),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(y.shape[1])
    ])

    model.compile(optimizer='adam', loss="mse", metrics=['mse'])
    model.fit(x_train, y_train, epochs=epochs, verbose=0)
    print("model score", model.evaluate(x_test, y_test))

    model_save_path = os.path.join('cloning_policy', envname + '.h5')
    print("cloning model saved to ", model_save_path)
    model.save(model_save_path)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_data_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    print('loading expert data')
    x, y = load_expert.load_expert_data(args.expert_data_file)
    print('loaded and learn')
    _clone_behavior(x, y, args.epochs, args.envname)


if __name__ == '__main__':
    main()
