from gym_wrapper import transform_env
import torch
import gym
from networks import networkV2
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
        action = probs.multinomial(num_samples=1)

        # return highest probability
        return action[0].item()  # TODO extract confidence


def play():
    path = 'models/test1/test1_0_a2c.pt'
    actions = 5
    agent = Agent(networkV2(actions))
    agent.load_model(path)

    env = gym.make('MsPacman-v0')
    env_ai = transform_env(env, 4)
    observation = env.reset()
    observation_ai = env_ai.reset()
    done = False

    while not done:
        env_ai = transform_env(env, 4)
        env.render()

        action_ai = agent.choose_action(torch.from_numpy(observation_ai))
        _, _, done, _ = env.step(action_ai)

        time.sleep(1 / 30)  # FPS

# TODO put everything in a class
# TODO add threads
# TODO split for user control and AI suggestions
# TODO embed environment into GUI
# TODO create and embed control matrix into GUI
# TODO embed graph with AI certainty into GUI
# TODO embed graph with player vs ai decision divergence ?
# TODO add "autopilot" option

if __name__ == '__main__':
    play()
