from agents import AgentA2C
from networks import networkV4
from workers import Worker
from gym_wrapper import make_env


def train():
    actions = 5
    workers_len = 10
    iterations = 10001

    # DeepMind DQN - 4
    stack = 4

    # Suggestion od mareka - 10
    steps = 20

    agent = AgentA2C("final", networkV4(actions), 0.99, 0.0005, 0.01, id=3)

    workers = []
    for id in range(workers_len):
        env = make_env('MsPacman-v0', stack)
        env.seed(id)
        w = Worker(id, env, agent, print_score=True)
        workers.append(w)

    episode = 27683
    path = 'models/final/final_2_10000_a2c.pt'
    agent.load_model(path)
    agent.learn(workers, iterations, steps, start_episode=episode)


if __name__ == '__main__':
    train()
