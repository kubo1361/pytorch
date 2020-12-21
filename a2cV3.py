from agents import AgentA2C
from networks import networkV1
from workers import Worker
from pong_wrapper import make_env
import torch as T


def train():
    actions = 5
    workers_len = 20
    iterations = 2000
    steps = 5

    agent = AgentA2C("a2c_Pac", networkV1(actions), 0.99, 0.001, 0.01)

    workers = []
    for id in range(workers_len):
        env = make_env('MsPacman-v0')
        env.seed(id)
        w = Worker(id, env, agent)
        workers.append(w)

    agent.learn(workers, iterations, steps)


def play():
    actions = 5
    workers_len = 1
    agent = AgentA2C("a2c_Pac_0_a2c.pt", networkV1(actions), 0.99, 0.001, 0.01)
    agent.load_model()

    workers = []
    env = make_env('MsPacman-v0')
    observation = env.reset()
    done = False
    while True:
        env.render()
        action = agent.choose_action(T.from_numpy(observation))
        observation, reward, done, info = env.step(action)
        if(done):
            env.reset()

if __name__ == '__main__':
    #train()
    play()