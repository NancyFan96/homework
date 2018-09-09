#!/usr/bin/env python

"""
Code to plot result figures.
Example usage:
    python plot.py

Author: Naijia Fan
"""
import matplotlib.pyplot as plt


def plot_1():

    epochs = [1, 3, 5, 7, 9, 11, 13]
    mean = [992.9256606827082, 3500.101101396331, 4350.861610917002, 4476.067357747206,
            3832.079715068327, 4491.220839019459, 4434.5819579225945]
    std = [98.56783110525221, 1152.1110585244494, 464.8945713422921, 259.3228763250365,
           1258.2538081014657, 744.6349768561219, 967.0909398729037]

    plt.figure(figsize=(10, 8))
    plt.plot(epochs, mean)
    plt.errorbar(epochs, mean, yerr=std, fmt='o')
    plt.title('Behavorial Cloning Rewards with Different Epochs', fontsize=11)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Reward')
    plt.xlim([0, 15])
    plt.legend(['Mean', 'Std'])
    plt.savefig('2-2.png')
    plt.show()


def plot_2():
    iters = range(1, 11)
    mean = [1512.4344919914515, 1207.1934207421843, 2622.5493232814893, 3769.397087119257,
            3779.1785373483167, 3720.4268652771307, 3782.5436357287967, 1695.9485864344504,
            3772.1398833895755 , 3784.5361999363477]
    std = [243.8018102481672, 169.4559711718996, 792.2196985176346, 3.22897032893531,
           4.341987597202986, 62.640423688618775, 2.7732957173867856, 116.32946914788599,
           3.582557223402759, 3.4321169281471904]
    expert = 3778.591392827052
    behaviral_cloning = 1353.4823545696288

    plt.figure(figsize=(10, 8))
    plt.plot(iters, mean)
    plt.errorbar(iters, mean, yerr=std, fmt='o')
    plt.title('DAgger Rewards with Different Dagger Iterations', fontsize=11)

    plt.xlabel('DAgger Iteration')
    plt.ylabel('Reward')
    plt.xlim([0, 12])

    plt.axhline(y=expert)
    plt.axhline(y=behaviral_cloning)
    plt.legend(['dagger mean', 'dagger std', 'expert', 'behavioral cloning'])
    plt.savefig('3-2.png')
    plt.show()


if __name__ == '__main__':
    plot_1()
    plot_2()
