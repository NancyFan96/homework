import pickle
import numpy as np
import matplotlib.pyplot as plt


def draw_episode_rewards(data, figName):
    ts = range(len(data))
    meanEps = np.full(100, -np.Inf)
    meanEps = np.concatenate([meanEps, np.array([np.mean(data[i-100:i]) for i in range(100, len(data))])])
    bestMeanEps = np.concatenate([np.array([-np.Inf]), np.array([np.max(meanEps[:i]) for i in range(1, len(data))])])

    plt.figure()
    plt.plot(ts, meanEps)
    plt.plot(ts, bestMeanEps)
    plt.title('Mean 100-Episode Reward and Best Mean 100-Episode Reward of Basic Q-Learning', fontsize=11)
    plt.xlabel('Time Step')
    plt.ylabel('Mean Reward')
    plt.legend(['Mean 100-Episode Reward', 'Best Mean 100-Episode Reward'])

    plt.savefig('doc/' + figName + '.png')
    plt.show()

def main():
    with open('dbc8b70a-5a49-4788-be9d-adb4fa673398.pkl', 'rb') as fh:
        data = pickle.loads(fh.read())
    fig1_name = 'p1q1-lander-test'
    draw_episode_rewards(data, fig1_name)

if __name__ == '__main__':
    main()
