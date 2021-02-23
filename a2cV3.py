from agents import AgentA2C
from networks import networkV2
from workers import Worker
from pong_wrapper import make_env


def train():
    actions = 5
    workers_len = 20
    iterations = 1001  # TODO zisti

    # DeepMind DQN - 4
    stack = 4

    # Suggestion od mareka - 10
    steps = 4

    agent = AgentA2C("test1", networkV2(actions), 0.99, 0.01, 0.01)

    workers = []
    for id in range(workers_len):
        env = make_env('MsPacman-v0', stack)
        env.seed(id)
        w = Worker(id, env, agent, print_score=True)
        workers.append(w)

    agent.learn(workers, iterations, steps)


# TODO vytvori sa uplne novy program do ktoreho zasadim ulozenu siet
"""
def play():
    actions = 5
    workers_len = 1
    agent = AgentA2C("a2c_Pac_0_a2c.pt", networkV1(actions), 0.99, 0.001, 0.01)
    agent.load_model()

    workers = []
    env = make_env('MsPacman-v0', 4)
    observation = env.reset()
    done = False
    while True:
        env.render()
        action = agent.choose_action(T.from_numpy(observation))
        observation, reward, done, info = env.step(action)
        if(done):
            env.reset()
"""

if __name__ == '__main__':
    train()
    # play()
