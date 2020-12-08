import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gym
import matplotlib.pyplot as plt
import gym.wrappers as w


PATH = "D:\PROJECTS\Bakalarka\model_data\modelV2"

class Net(nn.Module):
    def __init__(self, n_actions_, lr_=0.01):
        super(Net, self).__init__()
        self.model = nn.Sequential( 
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=[3, 3], stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=[3, 3], stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=[3, 3], stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Flatten(start_dim=1),

            nn.Linear(in_features=24 * 26 * 20, out_features=4096), #256 staci - max 512
            nn.Linear(in_features=4096, out_features=4096),
            nn.Linear(in_features=4096, out_features=n_actions_),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr_) # 1 optimizer
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')
        self.to(self.device)

    def forward(self, state):
        x = T.Tensor(state.float()).to(self.device)
        return self.model(x)

#TODO vykradni weightsInitXavier od mareka

class Agent(object):
    def __init__(self, lr_a, lr_c, gamma=0.99):
        self.gamma = gamma
        self.log_probs = None
        self.actor = Net(lr_=lr_a, n_actions_=9) #TODO merge - spolocna konvolucia
        self.critic = Net(lr_=lr_c, n_actions_=1) # akcie osekat na 5

    def choose_action(self, observation):
        distributions = T.distributions.Categorical(F.softmax(self.actor.forward(observation), dim=0)) #TODO upravit podla mareka
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

# pridaj entropiu - low prio
if __name__ == '__main__':
    agent = Agent(lr_a=0.01, lr_c=0.02, gamma=0.99) # 0.001

    env = w.FrameStack(w.GrayScaleObservation((gym.make('MsPacman-v0'))), 3)
    score_history = []
    n_episodes = 100

    for i in range(n_episodes):
        done = False
        score = 0
        observation = env.reset()
        observation = T.from_numpy(np.array(observation)).unsqueeze(0)

        while not done:
            env.render()
            action = agent.choose_action(observation)
            action.detach()
            observation_, reward, done, info = env.step(action.item())
            observation_ = T.from_numpy(np.array(observation_)).unsqueeze(0)
            score += reward
            agent.learn(observation, reward, observation_, done)
            observation = observation_

        print('episode', i, 'score %.3f' % score) 
        score_history.append(score)

    T.save(agent, PATH)

    x = [i + 1 for i in range(n_episodes)]
    fileName_ = 'pacman.png'
    plot_learning_curve(x, score_history, fileName_)


    """
    - pouzi marekove wrappere
    - mensiu siet
    - trenuj na pongu 
    - spojit actora a critica
    - 
    """