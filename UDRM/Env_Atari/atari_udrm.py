import random
import time
import os
import sys
from pathlib import Path
import numpy as np
import wandb
import torch
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
from utils import loss_fn, gamma_cal_loss
from quantilelosses import loss_evi

class textcolors:
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    DEFAULT = "\033[37m"
       
class UDRM():
    def __init__(
        self,
        env,
        network,
        n_quantiles=100,
        timesteps=2500000,
        obs_space_channel=4,
        kappa=1,
        replay_start_size=50000,
        replay_buffer_size=1000000,
        alpha_int=500,
        beta_int=5000,
        gamma=0.99,
        update_target_frequency=10000,
        batch_size=32,
        learning_rate=1e-4,
        adam_epsilon=1e-8,
        update_frequency=1,
        evi_coeff=1.0, 
        unc_lambda_ep=0.1,
        unc_lambda_al=0, 
        save_period=5000,
        logging=False,
        log_folder_details=None,
        train_file=None,
        wandb_url=None,
        seed=None,
        notes=None
    ):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.replay_start_size = replay_start_size
        self.replay_buffer_size = replay_buffer_size
        self.alpha_int = alpha_int
        self.beta_int = beta_int
        self.gamma = gamma
        self.update_target_frequency = update_target_frequency
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.update_frequency = update_frequency
        self.adam_epsilon = adam_epsilon
        self.logging = logging
        self.evi_coeff = evi_coeff
        self.unc_lambda_ep = unc_lambda_ep
        self.unc_lambda_al = unc_lambda_al
        self.obs_space_channel = obs_space_channel
        self.logger = []
        self.save_period = save_period
        self.timestep = 0
        self.train_file = train_file
        self.wandb_url = wandb_url

        self.n_quantiles = n_quantiles
        self.train_steps = timesteps
        # self.explore_steps = int(np.round(0.1 * self.train_steps)) # added

        self.env = env
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)
        self.seed = random.randint(0, 2e8) if seed is None else seed
        self.logger=None
        self.log_folder_details = log_folder_details + "-" + str(self.seed)
        
        set_global_seed(self.seed, self.env)

        self.network = network(self.device,self.env.observation_space, self.n_quantiles, self.env.action_space.n*self.n_quantiles,self.obs_space_channel).to(self.device)
        self.target_network = network(self.device,self.env.observation_space, self.n_quantiles, self.env.action_space.n*self.n_quantiles,self.obs_space_channel).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate, eps=self.adam_epsilon)

        self.loss = loss_fn
        self.kappa = kappa

        self.train_parameters = {
                'Notes':notes,
                'env':self.env.unwrapped.spec.id,
                'network':str(self.network),
                'device':str(self.device),
                'replay_start_size':self.replay_start_size,
                'replay_buffer_size':self.replay_buffer_size,
                'alpha_int':self.alpha_int,
                'beta_int':self.beta_int,
                'gamma':self.gamma,
                'update_target_frequency':self.update_target_frequency,
                'batch_size':self.batch_size,
                'learning_rate':self.learning_rate,
                'adam_epsilon':self.adam_epsilon,
                'update_frequency':self.update_frequency,
                'kappa':self.kappa,
                'train_file':self.train_file,
                'wandb_url': self.wandb_url,
                'executed_file':os.path.abspath(__file__),
                'n_quantiles':self.n_quantiles,
                'train_steps':self.train_steps,
                'max_episode_steps':self.env.spec.max_episode_steps,
                # 'explore_steps':self.explore_steps,
                'weight_scale':self.network.weight_scale,
                'evidence coeff':self.evi_coeff,
                'unc_lambda_ep':self.unc_lambda_ep,
                'unc_lambda_al':self.unc_lambda_al,
                'obs_space_channel':self.obs_space_channel,
                'save_period':self.save_period,
                'seed':self.seed
            }
            
        if self.logging:
            self.logger = Logger(self.log_folder_details, self.train_parameters, self.alpha_int)

        self.n_greedy_actions = 0

    def learn(self, verbose=False):
        pprint.pprint(self.train_parameters)

        # Initialize the state
        obs, info = self.env.reset(seed=self.seed)
        state = torch.as_tensor(np.array(obs))
        this_episode_time = 0
        score = 0
        t1 = time.time()

        actions_uncertain = 0
        actions_greedy = 0
        max_score = 0
        remembered = 0
        twice_remembered = 0
        cum_scores = []
        max_scores = []
        unc_history = []

        self.alpha = 1.0
        self.beta = 1.0
        self.threshold = 1.0

        moving_avg_window = 100
        beta_inc_factor = 1.05
        beta_dec_factor = 0.95

        with open(self.logger.log_folder + '/log_output.txt',"w") as f:

            for timestep in range(self.train_steps):
                is_training_ready = timestep >= self.replay_start_size

                # Select action
                action, least_uncertain, greedy, u_al, u_ep = self.act(timestep, state.to(self.device).float())
                u_total = u_al + u_ep
                wandb.log({"timestep": timestep, "unc_al": u_al})
                wandb.log({"timestep": timestep, "unc_ep": u_ep})
                wandb.log({"timestep": timestep, "unc_total": u_total})
                self.logger.add_scalar("unc_al", u_al, timestep)
                self.logger.add_scalar("unc_ep", u_ep, timestep)
                self.logger.add_scalar("unc_total", u_total, timestep)
                actions_uncertain += least_uncertain
                actions_greedy += greedy

                # Perform action in environments
                state_next, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # Store transition in replay buffer
                action = torch.as_tensor([action], dtype=torch.long)
                reward = torch.as_tensor([reward], dtype=torch.float)
                done = torch.as_tensor([done], dtype=torch.float)
                state_next = torch.as_tensor(np.array(state_next))
                self.replay_buffer.add(state, action, reward, state_next, done)

                if not is_training_ready:
                    unc_history.append(u_ep)
                    if len(unc_history) > moving_avg_window:
                        unc_history.pop(0)
                else:
                    if timestep == self.replay_start_size:
                        uncs_ep_tensor = torch.tensor(unc_history)
                        self.alpha = torch.quantile(uncs_ep_tensor, 0.99)
                        k = 2 # fraction of initial value
                        self.beta = torch.log(torch.tensor(k)) / self.train_steps 
                    else:
                        self.threshold = (self.alpha) * torch.exp(-(self.beta) * timestep)
                        remembered += 1
                        unc_history.append(u_ep)
                        if len(unc_history) > moving_avg_window:
                            unc_history.pop(0)
                        moving_avg_uncertainty = sum(unc_history) / len(unc_history)

                        if timestep % self.beta_int == 0 and len(unc_history) >= moving_avg_window:
                            if moving_avg_uncertainty > self.threshold:
                                self.beta *= beta_dec_factor
                            elif moving_avg_uncertainty < self.threshold:
                                self.beta *= beta_inc_factor
                            
                        if timestep % self.alpha_int == 0 and timestep > 0:
                            self.alpha = torch.quantile(torch.tensor(unc_history), 0.99)
                        
                        if (u_ep > self.threshold and reward.item() > 0):
                            self.replay_buffer.add(state, action, reward, state_next, done)
                            twice_remembered += 1
                        
                score += reward.item()
                this_episode_time += 1

                if done:        
                    if score > max_score:
                        textcolor = textcolors.BLUE
                        save_path = self.logger.log_folder + '/max_score.pth'
                        torch.save(self.network.state_dict(), save_path)
                        max_score = score
                    else:
                        textcolor = textcolors.DEFAULT

                    twice_remembered_pct = np.round(twice_remembered * 100 / remembered, 2) if remembered > 0 else 0
                    if verbose:
                        log_statement = textcolor + "Timestep: {}, Score: {}, Time: {} s, Actions: {}, Twice: {} ({}%), ALU: {} ({}%), AG: {} ({}%)".format(timestep, score, round(time.time() - t1, 3),
                                                                                                                                    this_episode_time,
                                                                                                                                    twice_remembered,
                                                                                                                                    twice_remembered_pct,
                                                                                                                                    actions_uncertain,
                                                                                                                                    round((actions_uncertain/this_episode_time)*100,2),
                                                                                                                                    actions_greedy,
                                                                                                                                    round((actions_greedy/this_episode_time)*100,2),
                                                                                                                                    )
                        print(log_statement)
                        f.write(log_statement.removeprefix(textcolor) + "\n")
                    non_greedy_fraction = 1-self.n_greedy_actions/this_episode_time
                    if self.logging:
                        wandb.log({"timestep": timestep, "Episode_score": score})
                        wandb.log({"timestep": timestep, "Non_greedy_fraction": non_greedy_fraction})
                        wandb.log({"timestep": timestep, "Twice_Remembered_pct": twice_remembered_pct})
                        self.logger.add_scalar('Episode_score', score, timestep)
                        self.logger.add_scalar('Non_greedy_fraction', non_greedy_fraction, timestep)
                        self.logger.add_scalar('Twice_Remembered_pct', twice_remembered_pct, timestep)
                        if timestep > self.replay_start_size:
                            wandb.log({"timestep": timestep, "Uncertainty_threshold": self.threshold})
                            self.logger.add_scalar('Uncertainty_threshold', self.threshold, timestep)
                    obs, info = self.env.reset(seed=self.seed)
                    state = torch.as_tensor(np.array(obs))
                    cum_scores.append(score)
                    max_scores.append(max_score)
                    score = 0
                    if self.logging:
                        max_q_start = self.get_max_q(state.to(self.device).float())
                        wandb.log({"timestep": timestep, "Q_at_start": max_q_start})
                        self.logger.add_scalar('Q_at_start', max_q_start, timestep)
                    t1 = time.time()
                    self.n_greedy_actions = 0
                    actions_uncertain = 0
                    actions_greedy = 0
                    this_episode_time = 0
                    remembered = 0
                    twice_remembered = 0
                else:
                    state = state_next

                if is_training_ready:

                    # Update main network
                    if timestep % self.update_frequency == 0:

                        # Sample a batch of transitions
                        transitions = self.replay_buffer.sample(self.batch_size, self.device)

                        # Train on selected batch
                        loss = self.train_step(transitions)
                        if done:
                            print(f"Loss: {loss:.3f}")
                        if self.logging:
                            wandb.log({"timestep": timestep, "loss": loss})
                            self.logger.add_scalar('Loss', loss, timestep)
                            
                    # Update target Q
                    if timestep % self.update_target_frequency == 0:
                        self.target_network.load_state_dict(self.network.state_dict())

                if (timestep+1) % self.save_period == 0:
                    self.save(timestep=timestep+1)
                    self.logger.save()

                self.timestep = timestep

            if self.logging:
                self.logger.save()
                self.save()
            
        return cum_scores, max_scores
        
    def train_step(self, transitions):
        states, actions, rewards, states_next, dones = transitions

        with torch.no_grad():
            target1,_ = self.target_network(states_next.float())
            target1 = target1.view(self.batch_size,self.env.action_space.n,self.n_quantiles)

        best_action_idx = torch.mean(target1,dim=2).max(1, True)[1].unsqueeze(2)
        q_value_target = target1.gather(1, best_action_idx.repeat(1,1,self.n_quantiles))

        # Calculate TD target
        td_target = (rewards.unsqueeze(2).repeat(1,1,self.n_quantiles) + (1 - dones.unsqueeze(2).repeat(1,1,self.n_quantiles)) * self.gamma * q_value_target).squeeze()

        out, evi = self.network(states.float())
        out = out.view(self.batch_size,self.env.action_space.n,self.n_quantiles) 

        gamma, v, alpha, beta = torch.split(evi, int(self.env.action_space.n*2*self.n_quantiles), dim=-1)
        gamma = gamma.view(self.batch_size, self.env.action_space.n, 2, self.n_quantiles).gather(1, actions.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 2, self.n_quantiles)).squeeze()
        v = v.view(self.batch_size, self.env.action_space.n, 2, self.n_quantiles).gather(1, actions.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 2, self.n_quantiles)).squeeze()
        alpha = alpha.view(self.batch_size, self.env.action_space.n, 2, self.n_quantiles).gather(1, actions.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 2, self.n_quantiles)).squeeze()
        beta = beta.view(self.batch_size, self.env.action_space.n, 2, self.n_quantiles).gather(1, actions.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 2, self.n_quantiles)).squeeze()

        q_value = out.gather(1, actions.unsqueeze(2).repeat(1,1,self.n_quantiles)).squeeze()
        
        assert q_value.shape == td_target.shape, f"Shape mismatch: {q_value.shape} vs {td_target.shape}"

        quantile_losses = self.loss(q_value, td_target, 0.5, self.kappa, self.device)
        loss_evidence = loss_evi(td_target, gamma, v, alpha, beta, self.evi_coeff)
        loss_gamma_cal = gamma_cal_loss(gamma,td_target,0.5,self.device)

        loss = quantile_losses + loss_evidence + loss_gamma_cal
        
        # Update weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def act(self, t, state):

        action, least_uncertain, greedy, u_al, u_ep = self.predict(t, state)        
        return action, least_uncertain, greedy, u_al, u_ep

    @torch.no_grad()
    def predict(self, t, state):
            
        out,evi = self.network(state)
        out = out.view(self.env.action_space.n,self.n_quantiles)
        action_means = torch.mean(out,dim=1)

        gamma, v, alpha, beta = torch.split(evi, int(self.env.action_space.n*2*self.n_quantiles), dim=-1)
        gamma = gamma.view(self.env.action_space.n,2,self.n_quantiles)
        v = v.view(self.env.action_space.n,2,self.n_quantiles)
        alpha = alpha.view(self.env.action_space.n,2,self.n_quantiles)
        beta = beta.view(self.env.action_space.n,2,self.n_quantiles)

        var = torch.sqrt((beta /(v*(alpha - 1))))
        
        gamma_0 = gamma[:,0,:]
        gamma_1 = gamma[:,1,:]
        var_0 = var[:,0,:]
        var_1 = var[:,1,:]

        global_aleatoric = torch.mean(torch.abs(gamma_1 - gamma_0),dim=1)
        global_epistemic = torch.mean(
            0.5*(torch.abs(2 * var_0) + torch.abs(2 * var_1)), 
            dim=1) + 1e-8

        # global_aleatoric = torch.mean(torch.abs(gamma[:,1,:] - gamma[:,0,:]),dim=1)
        # global_epistemic = torch.mean(0.5*(torch.abs((gamma[:,0,:]+var[:,0,:])-(gamma[:,0,:]-var[:,0,:])) + torch.abs((gamma[:,1,:]+var[:,1,:])-(gamma[:,1,:]-var[:,1,:]))),dim=1) + 1e-8

        # u_ep_act = 10000
        # u_al_act = 7

        # for al_i in range(len(global_aleatoric)):
        #     if torch.isinf(global_aleatoric[al_i]):
        #         global_aleatoric[al_i] = u_al_act

        # for ep_i in range(len(global_epistemic)):
        #     if torch.isinf(global_epistemic[ep_i]):
        #         global_epistemic[ep_i] = u_ep_act

        #action_means -= self.unc_lambda_al*(global_aleatoric)
        # action_uncertainties_cov = self.unc_lambda_ep*torch.diagflat(global_epistemic)
        cov_matrix = self.unc_lambda_ep * torch.diagflat(global_epistemic)
        action_uncertainties_cov = cov_matrix + torch.eye(len(global_epistemic)).to(self.device) * 1e-6
        
        samples = torch.distributions.multivariate_normal.MultivariateNormal(action_means,covariance_matrix=action_uncertainties_cov).sample()
        action_uncertain = (action_means - self.unc_lambda_ep*global_epistemic).argmax().item()
        action = samples.argmax().item()

        # if torch.isinf(global_epistemic[action]):
        #     non_inf_ep = global_epistemic[torch.isinf(global_epistemic) == False]
        #     if len(non_inf_ep) > 0:
        #         u_ep_act = max(non_inf_ep)
        # else:
        #     u_ep_act = global_epistemic[action]

        # if torch.isinf(global_aleatoric[action]):
        #     non_inf_al = global_aleatoric[torch.isinf(global_aleatoric) == False]
        #     if len(non_inf_al) > 0:
        #         u_al_act = max(non_inf_al)
        # else:
        #     u_al_act = global_aleatoric[action]

        if action == action_means.argmax().item():
            greedy = 1
            self.n_greedy_actions += 1
        else:
            greedy = 0

        if action_uncertain == action:
            least_uncertain = 1
        else:
            least_uncertain = 0

        u_al = global_aleatoric[action]
        u_ep = global_epistemic[action]
        
        return action, least_uncertain, greedy, u_al, u_ep

    @torch.no_grad()
    def get_max_q(self,state):
        net1,_ = self.network(state)
        net1 = net1.view(self.env.action_space.n,self.n_quantiles)
        action_means = torch.mean(net1,dim=1)
        max_q = action_means.max().item()
        return max_q

    def save(self,timestep=None):
        if not self.logging:
            raise NotImplementedError('Cannot save without log folder.')

        if timestep is not None:
            filename = 'network_' + str(timestep) + '.pth'
        else:
            filename = 'network.pth'

        save_dir = self.logger.log_folder + '/networks'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = save_dir + '/' + filename

        torch.save(self.network.state_dict(), save_path)

    def load(self,path):
        self.network.load_state_dict(torch.load(path,map_location='cpu'))
        self.target_network.load_state_dict(torch.load(path,map_location='cpu'))