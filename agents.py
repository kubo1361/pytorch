import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import write_to_file

class AgentA2C:
    def __init__(self, name, model, gamma=0.99, lr=0.0001, beta_entropy=0.001, value_loss_coef=0.5, id=0):
        # init vars
        self.model = model
        self.actions_count = model.actions_count
        self.gamma = gamma
        self.beta_entropy = beta_entropy
        self.value_loss_coef = value_loss_coef

        # device - define and cast
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('device: ', self.device)
        self.model.to(self.device)

        # define optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # create vars for tracking progress and identification
        self.average_score = []
        self.episodes = 0
        self.name = name
        self.id = id

    def choose_action(self, state):
        # add dimension (batch) to match (batch, layer, height, width), transfer to GPU
        state = state.unsqueeze(0).to(self.device).float()

        # we do not compute gradients when choosing actions, hence no_grad
        with torch.no_grad():
            outActor, _ = self.model(state)

        # transform output of forward pass, so probabilities will all add up to 1
        probs = F.softmax(outActor, dim=-1)

        # transfer to CPU after calculation
        probs = probs.cpu()

        # sort probabilities
        action = probs.multinomial(num_samples=1).detach() # CLARIFY naco to musim detachnut ked som nastavil no_grad ? (preco to nema aj softmax)

        # return highest probability
        return action[0].item()

    def learn(self, workers, iterations, steps, write=True):
        self.average_score = []
        best_avg = -100
        self.episodes = 0

        len_workers = len(workers)
        observations = []
        for worker in workers:
            observations.append(torch.from_numpy(worker.reset()).float())
        observations = torch.stack(observations).to(self.device)

        text = 'iteration,episode,score,step'
        iter_step = steps * len_workers

        for iteration in range(iterations):
            iter_critic_values = torch.zeros([steps, len_workers, 1])
            iter_actor_log_probs = torch.zeros([steps, len_workers, 1])
            iter_entropies = torch.zeros([steps, len_workers, 1])
            iter_rewards = torch.zeros([steps, len_workers, 1])
            iter_non_terminals = torch.ones([steps, len_workers, 1])

            for step in range(steps):
                step_actor, step_critic = self.model(observations)
                step_actor, step_critic = step_actor.cpu(), step_critic.cpu()

                step_probs = F.softmax(step_actor, dim=-1)
                step_log_probs = F.log_softmax(step_actor, dim=-1)
                step_entropies = (step_log_probs * step_probs).sum(1, keepdim=True)

                step_actions = step_probs.multinomial(num_samples=1).detach()
                step_log_probs_policy = step_log_probs.gather(1, step_actions)

                iter_critic_values[step] = step_critic
                iter_entropies[step] = step_entropies
                iter_actor_log_probs[step] = step_log_probs_policy

                step_rewards = torch.zeros([len_workers, 1])
                step_non_terminals = torch.ones([len_workers, 1], dtype=torch.int8)
                observations = []

                for worker in range(len_workers):
                    worker_observation, step_rewards[worker, 0], worker_terminal = workers[worker].step(step_actions[worker].item())

                    if worker_terminal:
                        step_non_terminals[worker, 0] = 0
                        worker_observation = workers[worker].reset()

                    observations.append(torch.from_numpy(worker_observation).float())

                observations = torch.stack(observations).to(self.device)

                iter_rewards[step] = step_rewards
                iter_non_terminals[step] = step_non_terminals

            with torch.no_grad():
                _, critic_values = self.model(observations)
                critic_values = critic_values.detach().cpu()

            advantages = torch.zeros([steps, len_workers, 1])
            for step in reversed(range(steps)):
                critic_values = iter_rewards[step] + self.gamma * critic_values * iter_non_terminals[step]
                advantages[step] = critic_values - iter_critic_values[step]

            advantages_detached = advantages.detach()

            value_loss = (advantages**2).mean()
            policy_loss = - (iter_actor_log_probs * advantages_detached).mean()
            entropy_loss = iter_entropies.mean()

            self.optimizer.zero_grad()
            loss = policy_loss + self.value_loss_coef * value_loss + self.beta_entropy * entropy_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            self.optimizer.step()

            if iteration % 25 == 0 and iteration > 0:
                avg = np.average(self.average_score[-100:])
                if avg > best_avg:
                    best_avg = avg
                    print('saving model, best score is ', best_avg)
                    torch.save(self.model.state_dict(), 'models/' + self.name + '_' + str(self.id) + '_a2c.pt')
                print(iteration, '\tepisodes: ', self.episodes, '\taverage score: ', avg)
                if write:
                    text += '\n' + str(iteration) + ',' + str(self.episodes) + ',' + str(avg) + ',' + str(iter_step * iteration)

                    if iteration % 1000 == 0:
                        self.average_score = self.average_score[-100:]
                        write_to_file(text, 'logs/' + self.name + '_' + str(self.id) + '_' + str(iteration) + '_a2c.txt')
                        torch.save(self.model.state_dict(), 'models/' + self.name + '_' + str(self.id) + '_' + str(iteration) + '_a2c.pt')

    def load_model(self):
        self.model.load_state_dict(torch.load('models/' + self.name))

    def save_model(self):
        torch.save(self.model.state_dict(), 'models/' + self.name + '_' + str(self.id) + '_a2c.pt')