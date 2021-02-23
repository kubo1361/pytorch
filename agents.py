import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import os
from utils import write_to_file

# TODO zisti co presne robi beta - 0.01 - https://arxiv.org/pdf/1602.01783.pdf deepmind a3c
# TODO zisti co presne robi gamma
# TODO nastavit learning rate
# TODO nastavit value loss


class AgentA2C:
    def __init__(self, name, model, gamma=0.99, lr=0.001, beta_entropy=0.01, critic_loss_coef=0.5, id=0):
        # init vars
        self.model = model
        self.actions_count = model.actions_count
        self.gamma = gamma
        self.beta_entropy = beta_entropy
        self.critic_loss_coef = critic_loss_coef

        # device - define and cast
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        print('Device: ', self.device)
        self.model.to(self.device)

        # define optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # create vars for tracking progress and identification
        self.average_score = []
        self.episodes = 0
        self.name = name
        self.id = id

        # create folders for models and logs
        self.model_path = 'models/' + self.name + '/'
        self.logs_path = 'logs/' + self.name + '/'

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.logs_path):
            os.makedirs(self.logs_path)

    def learn(self, workers, iterations, steps, write=True):
        # initial variables
        self.average_score = []
        best_avg = -100
        self.episodes = 0
        len_workers = len(workers)
        observations = []
        text = ""

        # initial observations
        for worker in workers:
            observations.append(torch.from_numpy(worker.reset()).float())
        observations = torch.stack(observations).to(self.device)

        for iteration in range(iterations):
            # iteration specific variables
            iter_critic_values = torch.zeros([steps, len_workers, 1])
            iter_actor_log_probs = torch.zeros([steps, len_workers, 1])
            iter_entropies = torch.zeros([steps, len_workers, 1])
            iter_rewards = torch.zeros([steps, len_workers, 1])
            iter_not_terminated = torch.ones([steps, len_workers, 1])

            for step in range(steps):
                # forward pass - actions and first critic values
                step_actor, step_critic = self.model(observations)
                step_actor, step_critic = step_actor.cpu(), step_critic.cpu()

                # step specific variables
                step_rewards = torch.zeros([len_workers, 1])
                step_not_terminated = torch.ones(
                    [len_workers, 1], dtype=torch.int8)
                observations = []

                # extract actions
                step_probs = F.softmax(step_actor, dim=-1)
                step_actions = step_probs.multinomial(num_samples=1).detach()

                # step entropy calculation
                step_log_probs = F.log_softmax(step_actor, dim=-1)
                step_entropies = (
                    step_log_probs * step_probs).sum(1, keepdim=True)
                step_log_probs_policy = step_log_probs.gather(1, step_actions)

                # update iteration steps
                iter_critic_values[step] = step_critic
                iter_entropies[step] = step_entropies
                iter_actor_log_probs[step] = step_log_probs_policy

                for worker in range(len_workers):
                    # Apply actions to workers enviroments
                    worker_observation, step_rewards[worker, 0], worker_terminated = workers[worker].step(
                        step_actions[worker].item())

                    # reset terminated workers
                    if worker_terminated:
                        step_not_terminated[worker, 0] = 0
                        worker_observation = workers[worker].reset()

                    # append new observations
                    observations.append(torch.from_numpy(
                        worker_observation).float())

                # update observations, rewards and terminated workers
                observations = torch.stack(observations).to(self.device)
                iter_rewards[step] = step_rewards
                iter_not_terminated[step] = step_not_terminated

            # forward pass - critic values after performing action
            with torch.no_grad():
                _, critic_values = self.model(observations)
                #  CLARIFY nemusi byt detach ked je to no grad ?
                critic_values = critic_values.detach().cpu()

            # compute advantage - we compute steps backwards
            # with their respective critic values for each step
            advantages = torch.zeros([steps, len_workers, 1])
            for step in reversed(range(steps)):
                critic_values = iter_rewards[step] + \
                    (self.gamma * critic_values * iter_not_terminated[step])

                advantages[step] = critic_values - iter_critic_values[step]

            # podla debugeru to nema require grad
            #  CLARIFY vsetko co vytvorim nove sa automaticky pripoji na graf ? :D
            advantages_detached = advantages.detach()

            # average reward for statistics
            average_reward = iter_rewards.mean().detach()  # CLARIFY treba to tiez detachnut ?

            # CLARIFY od mareka
            """
            # calculate losses
            critic_loss = (advantages**2).mean()
            actor_loss = - (iter_actor_log_probs * advantages_detached).mean()
            entropy_loss = iter_entropies.mean()

            # calculate final loss
            loss = actor_loss + \
                (self.critic_loss_coef * critic_loss) + \
                (self.beta_entropy * entropy_loss)
            """

            ###########################
            # calculate losses
            critic_loss = (advantages**2).mean() * self.critic_loss_coef
            actor_loss = - (iter_actor_log_probs * advantages_detached).mean()

            # CLARIFY entropia zvysuje loss, takze ju chceme tiez minimalizovat ?
            entropy_loss = - (iter_entropies.mean() * self.beta_entropy)

            # calculate final loss
            loss = actor_loss + critic_loss + entropy_loss
            #######################

            # clear gradients
            self.optimizer.zero_grad()

            # backward pass with our total loss https://www.datahubbs.com/two-headed-a2c-network-in-pytorch/
            loss.backward()

            # gradient clipping for exploding gradients https://towardsdatascience.com/what-is-gradient-clipping-b8e815cdfb48
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)

            # optimizer step
            self.optimizer.step()

            # stats
            if iteration % 10 == 0 and iteration > 0:

                # average for last 10 scores
                avg = np.average(self.average_score[-100:])

                # save model on new best average score
                if avg > best_avg:
                    best_avg = avg
                    print('Saving model, best score is: ', best_avg)
                    model_filename = (
                        self.model_path + self.name + '_' + str(self.id) + '_a2c.pt')
                    self.save_model(model_filename)

                # display informations
                print('iteration: ', iteration, '\tepisodes: ',
                      self.episodes, '\taverage score: ', avg)

                # write to file - log
                if write:
                    text += '\n' + str(iteration) + ',' + str(self.episodes) + \
                        ',' + str(avg) + ',' + str(average_reward.item()) + \
                        ',' + str(actor_loss.item()) + ',' + str(critic_loss.item()) + ',' + \
                        str(entropy_loss.item())

                    if iteration % 100 == 0:
                        self.average_score = self.average_score[-100:]

                        continuous_save_model_filename = (
                            self.model_path + self.name + '_' + str(self.id) + '_' + str(iteration) + '_a2c.pt')

                        continuous_save_logs_filename = (
                            self.logs_path + self.name + '_' + str(self.id) + '_' + str(iteration) + '_a2c.txt')

                        self.save_model(continuous_save_model_filename)
                        write_to_file(text, continuous_save_logs_filename)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)


# TODO pouzije sa pri vytvarani hracieho programu
"""
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
        return action[0].item()
"""
