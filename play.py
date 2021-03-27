from gym_wrapper import transform_observation
import torch
import gym
import numpy as np
from networks import networkV4
import torch.nn.functional as F
import time


class Agent:
    def __init__(self, model):
        # init vars
        self.model = model
        self.actions_count = model.actions_count

        # device - define and cast
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def choose_action(self, observation):
        # add dimension (batch) to match (batch, layer, height, width), transfer to GPU
        observation = observation.unsqueeze(0).to(self.device).float()

        # we do not compute gradients when choosing actions, hence no_grad
        with torch.no_grad():
            outActor, _ = self.model(observation)

        # transform output of forward pass, so probabilities will all add up to 1
        probs = F.softmax(outActor, dim=-1)

        # transfer to CPU after calculation
        probs = probs.cpu()

        # sort probabilities
        actions = probs.multinomial(num_samples=1)

        # return highest probability
        return actions[0].item()  # TODO extract confidence


def play():
    path = 'models/final/final_4_a2c.pt'
    actions = 5
    agent = Agent(networkV4(actions))
    agent.load_model(path)

    env = gym.make('MsPacman-v0')
    env.reset()

    done = False
    action_ai = 0
    observations = np.zeros((4, 80, 80), dtype=np.float32)
    while not done:
        for i in range(0, 4):
            env.render()
            obs, _, done, _ = env.step(action_ai)
            observations[i] = transform_observation(obs)

        action_ai = agent.choose_action(torch.from_numpy(observations))
        observations = np.zeros((4, 80, 80), dtype=np.float32)
        time.sleep(1 / 60)  # FPS


if __name__ == '__main__':
    play()
