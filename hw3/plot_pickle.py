import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def draw_episode_rewards(data, figName):
    """
    pickle.dump(episode_rewards, f, pickle.HIGHEST_PROTOCOL)

    :param data:
    :param figName:
    :return:
    """
    ts = np.array(range(len(data)))
    meanEps = np.full(100, -np.Inf)
    meanEps = np.concatenate([meanEps, np.array([np.mean(data[i-100:i]) for i in range(100, len(data))])])
    bestMeanEps = np.concatenate([np.array([-np.Inf]), np.array([np.max(meanEps[:i]) for i in range(1, len(data))])])

    plt.figure()
    plt.plot(ts, meanEps)
    plt.plot(ts, bestMeanEps)
    plt.title('Mean 100-Episode Reward of Basic Q-Learning', fontsize=11)
    plt.xlabel('Timesteps')
    plt.ylabel('Mean Episode Reward')
    plt.legend(['Mean', 'Best Mean'])
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 3))

    filename = 'doc/' + figName + '.png'
    print(filename)
    plt.savefig(filename)
    plt.show()


def draw_episode_rewards_by_timestamp(data, figName):
    """
    pickle.dump({
            'timestamp': [],
            'episode': [],
            'mean_episode_reward': [],
            'best_mean_episode_reward': []
          }, f, pickle.HIGHEST_PROTOCOL)

    :param data:
    :param figName:
    :return:
    """
    ts = data['timestamp']
    meanEps = data['mean_episode_reward']
    bestMeanEps = data['best_mean_episode_reward']

    plt.figure()
    plt.plot(ts, meanEps)
    plt.plot(ts, bestMeanEps)
    plt.title('Mean 100-Episode Reward of Basic Q-Learning', fontsize=11)
    plt.xlabel('Timesteps')
    plt.ylabel('Mean Episode Reward')
    plt.legend(['Mean', 'Best Mean'])
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 3))

    filename = 'doc/' + figName + '.png'
    print(filename)
    plt.savefig(filename)
    plt.show()


def get_data(pkls, names):
    datasets = []

    for pkl, name in zip(pkls, names):
        with open(pkl, 'rb') as fh:
            data_dict = pickle.loads(fh.read())
            data = pd.DataFrame.from_dict(data_dict)[["timestamp", "mean_episode_reward", "best_mean_episode_reward"]]
            data.insert(
                len(data.columns),
                'Name',
                name
            )

            datasets.append(data)

    return datasets


def plot_data(data, names, title):
    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)

    for name in names:
        sns.set()
        sns.lineplot(x="timestamp", y="best_mean_episode_reward", data=data[data["Name"] == name], label=name + ": best")
        sns.lineplot(x="timestamp", y="mean_episode_reward", data=data[data["Name"] == name], label=name + ": mean")

    plt.title(title)
    plt.xlabel('Timesteps')
    plt.ylabel('Mean Episode Reward')
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 3))

    import time
    filename = 'doc/' + names[0] + '_' + time.strftime("%d-%m-%Y_%H-%M-%S") + '.png'
    print(filename)
    plt.savefig(filename)
    plt.show()


def main():
    # a0d92d81-acd9-4076-ab3b-9bce9f02d5d8.pkl atari
    # dbc8b70a-5a49-4788-be9d-adb4fa673398.pkl lander
    # 2504a4ce-c448-43d0-b5a0-42e10a4ba8c2.pkl ram
    # a2df2ae9-4df5-4303-91fb-471412039e64.pkl atari timestampe
    # a806c937-b45c-4dcb-b7f6-dc8fb5b91fc6.pkl                  long
    #
    # 63f91c15-c20a-49fe-a12e-527898c818d4.pkl final


    # with open('dbc8b70a-5a49-4788-be9d-adb4fa673398.pkl', 'rb') as fh:
    #     data = pickle.loads(fh.read())
    # fig1_name = 'p1q1-lander-test-episode'

    # with open('2504a4ce-c448-43d0-b5a0-42e10a4ba8c2.pkl', 'rb') as fh:
    #     data = pickle.loads(fh.read())
    # fig1_name = 'p1q1-ram-test'

    # with open('a0d92d81-acd9-4076-ab3b-9bce9f02d5d8.pkl', 'rb') as fh:
    #     data = pickle.loads(fh.read())
    # fig1_name = 'p1q1-atari-test-episode'

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--title', '-t', type=str, default="Mean 100-Episode Reward of Basic Q-Learning")
    parser.add_argument('--file_name', '-f', type=str, nargs='*', required=True)
    parser.add_argument('--exp_name', '-e', type=str, nargs='*')
    args = parser.parse_args()

    if isinstance(args.file_name, list):
        files = args.file_name
    else:
        files = [args.file_name]

    if not args.exp_name:
        import os
        names = [os.path.split(file)[-1][:5] for file in files]
    else:
        names = args.exp_name

    data = get_data(files, names)
    plot_data(data, names, args.title)


if __name__ == '__main__':
    main()
