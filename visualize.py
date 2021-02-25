import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd


def load_file(filepath):
    colnames = ['iteration', 'episode', 'score', 'reward',
                'actor_loss', 'critic_loss', 'entropy_loss']
    data = pd.read_csv(filepath, names=colnames, dtype=np.float)
    return data


def plot_results(data):
    ax0 = plt.subplot(3, 2, 1)
    ax0.plot(data.episode, data.score)
    plt.title('Score')

    ax1 = plt.subplot(3, 2, 2)
    ax1.plot(data.episode, data.reward)
    plt.title('Reward')

    ax2 = plt.subplot(3, 2, 3)
    ax2.plot(data.episode, data.actor_loss)
    plt.title('Actor loss')

    ax3 = plt.subplot(3, 2, 4)
    ax3.plot(data.episode, data.critic_loss)
    plt.title('Critic loss')

    ax4 = plt.subplot(3, 2, 5)
    ax4.plot(data.episode, data.entropy_loss)
    plt.title('Entropy loss')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    data = load_file('logs/test2/test2_0_3000_a2c.txt')
    plot_results(data)
    # TODO COMPARE
