from agents import AgentA2C
from networks import networkV1, networkV2, networkV3
from workers import Worker
from gym_wrapper import make_env


def train():
    actions = 5
    workers_len = 20
    iterations = 4001

    # DeepMind DQN - 4
    stack = 4

    # Suggestion od mareka - 10
    steps = 10

    agent = AgentA2C("test5", networkV3(actions), 0.99, 0.0005, 0.01)

    workers = []
    for id in range(workers_len):
        env = make_env('MsPacman-v0', stack)
        env.seed(id)
        w = Worker(id, env, agent, print_score=True)
        workers.append(w)

    agent.learn(workers, iterations, steps)


if __name__ == '__main__':
    train()
