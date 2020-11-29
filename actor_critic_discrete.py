import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gym
import matplotlib.pyplot as plt
from gym import wrappers

class Net(nn.Module):
    def __init__(self, n_actions_, lr_=0.01):
        super(Net, self).__init__()
        self.model = nn.Sequential( # Network
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=8 * 52 * 40, out_features=1024),
            nn.Linear(in_features=1024, out_features=1024),
            nn.Linear(in_features=1024, out_features=n_actions_),
            nn.Softmax(dim=1),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr_)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')
        self.to(self.device)

    def forward(self, state):
        x = T.Tensor(state.float()).to(self.device)
        return self.model(x)



class Agent(object):
    def __init__(self, lr_a, lr_c, gamma=0.99):
        self.gamma = gamma
        self.log_probs = None
        self.actor = Net(lr_=lr_a, n_actions_=9) #Separate instances for actor and critic
        self.critic = Net(lr_=lr_c, n_actions_=1)

    def choose_action(self, observation):
        distributions = T.distributions.Categorical(self.actor.forward(observation))
        action = distributions.sample()
        self.log_probs = distributions.log_prob(action)
        return action

    def learn(self, state, reward, next_state, done):
        critic_value = self.critic.forward(state)
        critic_value_next = self.critic.forward(next_state)

        advantage = (reward + (self.gamma * critic_value_next * (1 - int(done))) - critic_value)
        
        critic_loss = advantage.pow(2).mean()
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        actor_loss = -(self.log_probs * advantage.detach())
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()
        


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100):(i + 1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)


if __name__ == '__main__':
    agent = Agent(lr_a=0.00001, lr_c=0.0005, gamma=0.99)

    env = gym.make('MsPacman-v0')
    score_history = []
    n_episodes = 1500

    for i in range(n_episodes):
        done = False
        score = 0
        observation = env.reset()
        observation =  observation.transpose(2,0,1)
        observation =  T.from_numpy(observation).unsqueeze(0)

        while not done:
            env.render()
            action = agent.choose_action(observation)
            action.detach()
            observation_, reward, done, info = env.step(action.item())
            observation_ =  observation_.transpose(2,0,1)
            observation_ =  T.from_numpy(observation_).unsqueeze(0)
            score += reward
            agent.learn(observation, reward, observation_, done)
            observation = observation_

        print('episode', i, 'score %.3f' % score) 
        score_history.append(score)

    x = [i + 1 for i in range(n_episodes)]
    fileName_ = 'pacman.png'
    plot_learning_curve(x, score_history, fileName_)