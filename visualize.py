import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd


def load_file(filepath):
    colnames = ['iteration', 'episode', 'steps', 'score', 'reward',
                'actor_loss', 'critic_loss', 'entropy_loss']
    data = pd.read_csv(filepath, names=colnames, dtype=np.float)
    return data


def plot_results(data, data2):
    ax0 = plt.subplot(4, 2, 1)
    ax0.plot(data.episode, data.score)
    ax0.plot(data2.episode, data2.score)
    plt.title('Average score per 100 episodes')

    ax1 = plt.subplot(4, 2, 2)
    ax1.plot(data.episode, data.reward)
    ax1.plot(data2.episode, data2.reward)
    plt.title('Reward')

    ax2 = plt.subplot(4, 2, 3)
    ax2.plot(data.episode, data.actor_loss)
    ax2.plot(data2.episode, data2.actor_loss)
    plt.title('Actor loss')

    ax3 = plt.subplot(4, 2, 4)
    ax3.plot(data.episode, data.critic_loss)
    ax3.plot(data2.episode, data2.critic_loss)
    plt.title('Critic loss')

    ax4 = plt.subplot(4, 2, 5)
    ax4.plot(data.episode, data.entropy_loss)
    ax4.plot(data2.episode, data2.entropy_loss)
    plt.title('Entropy loss')

    ax5 = plt.subplot(4, 2, 6)
    ax5.plot(data.episode, data.steps)
    ax5.plot(data2.episode, data2.steps)
    plt.title('Steps per episode')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    data = load_file('logs/finalfinal/final_0_519000_a2c.txt')
    data2 = load_file('logs/finalfinal/final_0_519000_a2c.txt')
    plot_results(data, data2)
