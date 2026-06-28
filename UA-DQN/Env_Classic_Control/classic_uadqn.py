"""
Baseline Script: UA-DQN
Status: ADAPTED BASELINE (STRUCTURAL MODIFICATIONS MADE)

Original Authors: Stutts et al., Clements et al.
Source: https://github.com/acstutts/CEQR-DQN/tree/main
License: MIT

Copyright (c) 2020 IndustAI
Copyright (c) 2024 Alex Christopher Stutts
Copyright (c) 2026 Sheeraja Rajakrishnan (Modifications for Environmental Adaptation)

Modifications:
- This script builds directly upon Stutts et al. and provides an independent re-implementation of the baseline functionality originally introduced in Clements et al.
- Architectural an hyperparameter changes were made to the baseline model, for environmental adaptation.
"""

import random
import time
import os
import sys
from pathlib import Path
import numpy as np
import wandb
import torch
import torch.nn.functional as F
import torch.optim as optim
import pprint as pprint

# 1. Get the current directory
current_dir = Path(__file__).resolve().parent

# 2. Get the parent directory
parent_dir = current_dir.parent

# 3. Add parent directory to sys.path if it isn't already there
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from replay_buffer import ReplayBuffer
from logger import Logger
from utils import set_global_seed
from utils import quantile_huber_loss

class textcolors:
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    DEFAULT = "\033[37m"

class UADQN:
    def __init__(
        self,
        env,
        network,
        n_quantiles=50,
        timesteps=50000,
        obs_space_channel=4,
        kappa=1,
        replay_start_size=50,
        replay_buffer_size=50000,
        weight_scale=3,
        noise_scale=0.1,
        gamma=0.99,
        tau=0.005,
        batch_size=32,
        learning_rate=1e-3,
        adam_epsilon=1e-8,
        update_frequency=1,
        epistemic_factor=1.0,
        aleatoric_factor=1.0,
        log_folder_details=None,
        train_file=None,
        wandb_url=None,
        logging=False,
        notes=None,
        save_period=250000,
        seed=None,
        biased_aleatoric=False,
        render=False,
    ):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Agent parameters
        self.replay_start_size = replay_start_size
        self.replay_buffer_size = replay_buffer_size
        self.weight_scale = weight_scale
        self.noise_scale = noise_scale
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.update_frequency = update_frequency
        self.adam_epsilon = adam_epsilon
        self.epistemic_factor = epistemic_factor
        self.aleatoric_factor = aleatoric_factor
        self.obs_space_channel = obs_space_channel
        self.logger = []
        self.biased_aleatoric = biased_aleatoric
        self.logging = logging
        self.save_period = save_period
        self.render = render
        self.notes = notes
        self.timestep = 0
        self.train_file = train_file
        self.wandb_url = wandb_url

        self.n_quantiles = n_quantiles
        self.train_steps = timesteps

        # Initialize agent
        self.env = env
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)
        self.seed = random.randint(0, 1e6) if seed is None else seed
        self.logger = None
        self.log_folder_details = log_folder_details + "-" + str(self.seed)
        
        torch.manual_seed(self.seed)
        self.env.reset(seed=self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        self.kappa = kappa
        self.loss = quantile_huber_loss
        
        n_outputs = self.env.action_space.n*self.n_quantiles
        self.network = network(self.env.observation_space, n_outputs).to(self.device)
        self.target_network = network(self.env.observation_space, n_outputs).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())

        # Initialize anchored networks
        self.posterior1 = network(self.env.observation_space, n_outputs, weight_scale=weight_scale).to(self.device)
        self.posterior2 = network(self.env.observation_space, n_outputs, weight_scale=weight_scale).to(self.device)
        self.anchor1 = [p.data.clone() for p in list(self.posterior1.parameters())]
        self.anchor2 = [p.data.clone() for p in list(self.posterior2.parameters())]

        # Initialize optimizer
        params = list(self.network.parameters()) + list(self.posterior1.parameters()) + list(self.posterior2.parameters())
        self.optimizer = optim.Adam(params, lr=self.learning_rate, eps=self.adam_epsilon)

        # Figure out what the scale of the prior is from empirical std of network weights
        with torch.no_grad():
            std_list = []
            for i, p in enumerate(self.posterior1.parameters()):
                std_list.append(torch.std(p))
        self.prior_scale = torch.stack(std_list).mean().item()

        # Parameters to save to log file
        self.train_parameters = {
                'Notes': notes,
                'env': self.env.unwrapped.spec.id,
                'network': str(self.network),
                'device':str(self.device),
                'replay_start_size': self.replay_start_size,
                'replay_buffer_size': self.replay_buffer_size,
                'gamma': self.gamma,
                'tau': self.tau,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'adam_epsilon': self.adam_epsilon,
                'update_frequency': self.update_frequency,
                'kappa':self.kappa,
                'train_file':self.train_file,
                'wandb_url': self.wandb_url,
                'executed_file':os.path.abspath(__file__),
                'n_quantiles': self.n_quantiles,
                'train_steps':self.train_steps,
                'max_episode_steps':self.env.spec.max_episode_steps,
                'weight_scale': self.weight_scale,
                'noise_scale': self.noise_scale,
                'unc_lambda_ep': self.epistemic_factor,
                'unc_lambda_al': self.aleatoric_factor,
                'biased_aleatoric': self.biased_aleatoric,
                'obs_space_channel':self.obs_space_channel,
                'save_period':self.save_period,
                'seed': self.seed
            }
            
        if self.logging:
            self.logger = Logger(self.log_folder_details, self.train_parameters)
        
        self.non_greedy_actions = 0

    def learn(self, verbose=False):
        pprint.pprint(self.train_parameters)

        # Initialize the state
        state, info = self.env.reset(seed=self.seed)
        state = torch.as_tensor(np.array(state))
        self.timestep = 0
        self.this_episode_time = 0
        self.n_events = 0  # Number of times an important event is flagged in the info
        score = 0
        true_score = 0
        t1 = time.time()
        
        max_score = 0
        cum_scores = []
        max_scores = []
        
        with open(self.logger.log_folder + '/log_output.txt',"w") as f:
            
            for timestep in range(self.train_steps):
                is_training_ready = timestep >= self.replay_start_size

                if self.render:
                    self.env.render()

                # Select action
                action = self.act(timestep, state.to(self.device).float())

                # Perform action in environment
                state_next, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                if (info == "The agent fell!") and self.logging:  # For gridworld experiments
                    self.n_events += 1
                    self.logger.add_scalar('Agent falls', self.n_events, timestep)

                # Store transition in replay buffer
                action = torch.as_tensor([action], dtype=torch.long)
                reward = torch.as_tensor([reward], dtype=torch.float)
                done = torch.as_tensor([done], dtype=torch.float)
                state_next = torch.as_tensor(state_next)
                self.replay_buffer.add(state, action, reward, state_next, done)

                score += reward.item()
                self.this_episode_time += 1

                if done:
                    if "episode" in info:
                        true_score = info["episode"]["r"][0]
                        
                    if true_score > max_score:
                        textcolor = textcolors.BLUE
                        save_path = self.logger.log_folder + '/max_score.pth'
                        torch.save(self.network.state_dict(), save_path)
                        max_score = true_score
                    else:
                        textcolor = textcolors.DEFAULT

                    if verbose:
                        log_statement = textcolor + "Timestep: {}, Score: {}, Time: {} s".format(timestep, score, round(time.time() - t1, 3))
                        print(log_statement)
                        f.write(log_statement.removeprefix(textcolor) + "\n")

                    non_greedy_fraction = self.non_greedy_actions/self.this_episode_time
                    if self.logging:
                        wandb.log({"timestep": timestep, "Episode_score": true_score})
                        wandb.log({"timestep": timestep, "Non_greedy_fraction": non_greedy_fraction})
                        self.logger.add_scalar('Episode_score', true_score, timestep)
                        self.logger.add_scalar('Non Greedy Fraction', non_greedy_fraction, timestep)

                    # Reinitialize the state
                    state, info = self.env.reset(seed=self.seed)
                    state = torch.as_tensor(np.array(state))
                    cum_scores.append(score)
                    max_scores.append(max_score)
                    score = 0
                    true_score = 0
                    if self.logging:
                        max_q_start = self.predict(state.to(self.device).float())
                        wandb.log({"timestep": timestep, "Q_at_start": max_q_start})
                        self.logger.add_scalar('Q_at_start', max_q_start, timestep)
                    self.non_greedy_actions = 0
                    self.this_episode_time = 0
                    t1 = time.time()
                else:
                    state = state_next

                if is_training_ready:

                    # Update main network
                    if timestep % self.update_frequency == 0:

                        # Sample batch of transitions
                        transitions = self.replay_buffer.sample(self.batch_size, self.device)

                        # Train on selected batch
                        loss, anchor_loss = self.train_step(transitions)
                        if done:
                            print(f"Loss: {loss:.3f}")
                        if self.logging:
                            wandb.log({"timestep": timestep, "loss": loss})
                            self.logger.add_scalar('Loss', loss, timestep)
                            # self.logger.add_scalar('Anchor Loss', anchor_loss, timestep)

                    # Update target Q
                    with torch.no_grad():
                        for target_param, param in zip(self.target_network.parameters(), self.network.parameters()):
                            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

                    # if timestep % self.update_target_frequency == 0:
                    #     self.target_network.load_state_dict(self.network.state_dict())

                if (timestep+1) % self.save_period == 0:
                    self.save(timestep=timestep+1)
                    self.logger.save()

                self.timestep = timestep

        if self.logging:
            self.logger.save()
            self.save()

        if self.render:
            self.env.close()
        
        return cum_scores, max_scores

    def train_step(self, transitions):
        states, actions, rewards, states_next, dones = transitions

        # Calculate target Q
        with torch.no_grad():
            target = self.target_network(states_next.float())
            target = target.view(self.batch_size, self.env.action_space.n, self.n_quantiles)

        # Calculate max of target Q values
        best_action_idx = torch.mean(target, dim=2).max(1, True)[1].unsqueeze(2)
        q_value_target = target.gather(1, best_action_idx.repeat(1, 1, self.n_quantiles))

        # Calculate TD target
        rewards = rewards.unsqueeze(2).repeat(1, 1, self.n_quantiles)
        dones = dones.unsqueeze(2).repeat(1, 1, self.n_quantiles)
        td_target = rewards + (1 - dones) * self.gamma * q_value_target

        # Calculate Q value of actions played
        outputs = self.network(states.float())
        outputs = outputs.view(self.batch_size, self.env.action_space.n, self.n_quantiles)
        actions = actions.unsqueeze(2).repeat(1, 1, self.n_quantiles)
        q_value = outputs.gather(1, actions)

        # TD loss for main network
        loss = self.loss(q_value.squeeze(), td_target.squeeze(), self.device, kappa=self.kappa)

        # Calculate predictions of posterior networks
        posterior1 = self.posterior1(states.float())
        posterior1 = posterior1.view(self.batch_size, self.env.action_space.n, self.n_quantiles)
        posterior1 = posterior1.gather(1, actions)

        posterior2 = self.posterior2(states.float())
        posterior2 = posterior2.view(self.batch_size, self.env.action_space.n, self.n_quantiles)
        posterior2 = posterior2.gather(1, actions)

        # Regression loss for the posterior networks
        loss_posterior1 = self.loss(posterior1.squeeze(), td_target.squeeze(), self.device, kappa=self.kappa)
        loss_posterior2 = self.loss(posterior2.squeeze(), td_target.squeeze(), self.device, kappa=self.kappa)
        loss += loss_posterior1 + loss_posterior2

        # Anchor loss for the posterior networks
        anchor_loss = self.calc_anchor_loss()        
        loss += anchor_loss

        # Update weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), anchor_loss.mean().item()

    def calc_anchor_loss(self):
        """
        Returns loss from anchoring
        """

        diff1 = []
        for i, p in enumerate(self.posterior1.parameters()):
            diff1.append(torch.sum((p - self.anchor1[i])**2))
        diff1 = torch.stack(diff1).sum()

        diff2 = []
        for i, p in enumerate(self.posterior2.parameters()):
            diff2.append(torch.sum((p-self.anchor2[i])**2))
        diff2 = torch.stack(diff2).sum()

        diff = diff1 + diff2

        num_data = np.min([self.timestep, self.replay_buffer_size])
        anchor_loss = self.noise_scale**2*diff/(self.prior_scale**2*num_data)

        return anchor_loss

    @torch.no_grad()
    def get_q(self, state):
        net = self.network(state).view(self.env.action_space.n, self.n_quantiles)
        action_means = torch.mean(net, dim=1)
        q = action_means
        return q

    @torch.no_grad()
    def act(self, t, state):
        """
        Returns action to be performed using Thompson sampling
        with estimates provided by the two posterior networks
        """

        net = self.network(state).view(self.env.action_space.n, self.n_quantiles)

        posterior1 = self.posterior1(state).view(self.env.action_space.n, self.n_quantiles)
        posterior2 = self.posterior2(state).view(self.env.action_space.n, self.n_quantiles)

        mean_action_values = torch.mean(net, dim=1)

        # Calculate aleatoric uncertainty
        if self.biased_aleatoric:
            uncertainties_aleatoric = torch.std(net, dim=1)
        else:
            covariance = torch.mean((posterior1-torch.mean(posterior1))*(posterior2-torch.mean(posterior2)), dim=1)
            uncertainties_aleatoric = torch.sqrt(F.relu(covariance))

        # Aleatoric-adjusted Q values
        aleatoric_factor = torch.as_tensor(self.aleatoric_factor, dtype=torch.float).to(self.device)
        adjusted_action_values = mean_action_values - aleatoric_factor*uncertainties_aleatoric

        # Calculate epistemic uncertainty
        uncertainties_epistemic = torch.mean((posterior1-posterior2)**2, dim=1)/2 + 1e-8
        epistemic_factor = torch.as_tensor(self.epistemic_factor, dtype=torch.float).to(self.device)**2
        uncertainties_cov = epistemic_factor*torch.diagflat(uncertainties_epistemic)

        # Draw samples using Thompson sampling
        epistemic_distrib = torch.distributions.multivariate_normal.MultivariateNormal
        samples = epistemic_distrib(adjusted_action_values, covariance_matrix=uncertainties_cov).sample()
        action = samples.argmax().item()

        if action != mean_action_values.argmax().item():
            self.non_greedy_actions += 1
        
        return action

    @torch.no_grad()
    def predict(self, state):
        """
        Returns action with the highest Q-value
        """
        net = self.network(state).view(self.env.action_space.n, self.n_quantiles)
        mean_action_values = torch.mean(net, dim=1)
        action = mean_action_values.argmax().item()

        return action

    def save(self, timestep=None):
        """
        Saves network weights
        """
        if timestep is not None:
            filename = 'network_' + str(timestep) + '.pth'
            filename_posterior1 = 'network_posterior1_' + str(timestep) + '.pth'
            filename_posterior2 = 'network_posterior2_' + str(timestep) + '.pth'
        else:
            filename = 'network.pth'
            filename_posterior1 = 'network_posterior1.pth'
            filename_posterior2 = 'network_posterior2.pth'

        save_dir = self.logger.log_folder + '/networks'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        save_path = save_dir + '/' + filename
        save_path_posterior1 = save_dir + '/' + filename_posterior1
        save_path_posterior2 = save_dir + '/' + filename_posterior2

        torch.save(self.network.state_dict(), save_path)
        torch.save(self.posterior1.state_dict(), save_path_posterior1)
        torch.save(self.posterior2.state_dict(), save_path_posterior2)

    def load(self, path):
        """
        Loads network weights
        """
        self.network.load_state_dict(torch.load(path + 'network.pth', map_location='cpu'))
        self.posterior1.load_state_dict(torch.load(path + 'network_posterior1.pth', map_location='cpu'))
        self.posterior2.load_state_dict(torch.load(path + 'network_posterior2.pth', map_location='cpu'))
