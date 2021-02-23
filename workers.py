def reward_func(r):  # CLARIFY
    if r > 1:
        return 1
    elif r < -1:
        return -1
    return r


class Worker:
    def __init__(self, id, env, agent, print_score=False, reward_function=reward_func):
        self.id = id
        self.env = env

        self.print_score = print_score
        self.episode = 1
        self.observation = None
        self.score = 0
        self.agent = agent
        self.reward_function = reward_function

    def reset(self):
        if self.print_score and self.episode % 10 == 0:
            print('worker: ', self.id, '\tepisode: ',
                  self.episode, '\tscore: ', self.score)
        self.agent.average_score.append(self.score)
        self.agent.episodes += 1
        self.observation = self.env.reset()
        self.episode += 1
        self.score = 0
        return self.observation

    def step(self, action):
        self.observation, reward, terminate, _ = self.env.step(action)
        self.score += reward

        reward = self.reward_function(reward)

        return self.observation, reward, terminate
