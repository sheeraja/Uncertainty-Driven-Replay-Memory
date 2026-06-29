import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def init_weights(m, gain):
    if (type(m) == nn.Linear) | (type(m) == nn.Conv2d):
        nn.init.orthogonal_(m.weight, gain)
        nn.init.zeros_(m.bias)

class CNN_CEQRDQN(nn.Module):
    def __init__(self, device, observation_space, n_quantiles, n_outputs_1, obs_space_channel, weight_scale=np.sqrt(2)):
        super().__init__()
        self.device = device
        self.n_quantiles = n_quantiles
        self.obs_space_channel = obs_space_channel
        self.n_actions = int(n_outputs_1 / n_quantiles)
        self.weight_scale = np.sqrt(2)

        if len(observation_space.shape) != 3:
            raise NotImplementedError

        self.conv = nn.Sequential(
            nn.Conv2d(self.obs_space_channel,16,3,stride=1,padding=1),
            nn.PReLU(num_parameters=16),
        )

        self.out_hidden = nn.Sequential(
            nn.Linear(in_features=1600, out_features=1600),
            nn.ReLU()
        )
        
        self.output = nn.Sequential(
            nn.Linear(1600, n_outputs_1),
        )

        self.NIG = nn.Sequential(
            nn.Linear(1600, 4*self.n_actions*2*self.n_quantiles), # = 4 * 2 * n_outputs_1
        )

        self.conv.apply(lambda x: init_weights(x, self.weight_scale))
        self.out_hidden.apply(lambda x: init_weights(x, self.weight_scale))
        self.output.apply(lambda x: init_weights(x, self.weight_scale))
        self.NIG.apply(lambda x: init_weights(x, self.weight_scale))

    def evi_split(self, out):
        mu, logv, logalpha, logbeta = torch.split(out, self.n_actions*2*self.n_quantiles, dim=-1)

        v = F.softplus(logv)
        alpha = F.softplus(logalpha) + 1
        beta = F.softplus(logbeta)
        return torch.concat([mu, v, alpha, beta], axis=-1)

    def forward(self, obs):
        if len(obs.shape) != 4:
            obs = obs.unsqueeze(0)
        obs = obs.permute(0,3,1,2)
        obs = self.conv(obs)
        obs = obs.contiguous().view(obs.size(0), -1)

        hidden = self.out_hidden(obs)
        output = self.output(hidden)
        G_evi = self.NIG(hidden)
        G_evi = self.evi_split(G_evi)

        return output, G_evi

class MLP_CEQRDQN(torch.nn.Module):
    def __init__(self, device, observation_space, n_quantiles, n_outputs_1, obs_space_channel, hiddens=[64, 64], weight_scale=3, **kwargs):
        super().__init__()
        self.device = device
        self.n_quantiles = n_quantiles
        self.obs_space_channel = obs_space_channel
        self.n_actions = int(n_outputs_1 / n_quantiles)
        self.weight_scale = np.sqrt(2)

        if len(observation_space.shape) != 1:
            raise NotImplementedError
        else:
            n_inputs = observation_space.shape[0]

        self.hidden1 = nn.Sequential(
            nn.Linear(in_features=n_inputs, out_features=64),
            nn.ReLU()
        )

        self.hidden2 = nn.Sequential(
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU()
        )        
        
        self.output = nn.Sequential(
            nn.Linear(64, n_outputs_1),
        )

        self.NIG = nn.Sequential(
            nn.Linear(64, 4*self.n_actions*2*self.n_quantiles), # = 4 * 2 * n_outputs_1
        )

        self.hidden1.apply(lambda x: init_weights(x, self.weight_scale))
        self.hidden2.apply(lambda x: init_weights(x, self.weight_scale))
        self.output.apply(lambda x: init_weights(x, self.weight_scale))
        self.NIG.apply(lambda x: init_weights(x, self.weight_scale))

    def evi_split(self, out):
        mu, logv, logalpha, logbeta = torch.split(out, self.n_actions*2*self.n_quantiles, dim=-1)

        v = F.softplus(logv)
        alpha = F.softplus(logalpha) + 1
        beta = F.softplus(logbeta)
        return torch.concat([mu, v, alpha, beta], axis=-1)

    def forward(self, obs):
        obs = self.hidden1(obs)
        hidden = self.hidden2(obs)
        output = self.output(hidden)
        G_evi = self.NIG(hidden)
        G_evi = self.evi_split(G_evi)

        return output, G_evi

class MLP_Toy_CEQRDQN(torch.nn.Module):
    def __init__(self, device, observation_space, n_quantiles, n_outputs_1, obs_space_channel, hiddens=[12, 8], weight_scale=3, **kwargs):
        super().__init__()
        self.device = device
        self.n_quantiles = n_quantiles
        self.obs_space_channel = obs_space_channel
        self.n_actions = int(n_outputs_1 / n_quantiles)
        self.weight_scale = np.sqrt(2)

        if len(observation_space.shape) != 0:
            raise NotImplementedError
        else:
            n_inputs = observation_space.n

        self.hidden1 = nn.Sequential(
            nn.Linear(in_features=n_inputs, out_features=12),
            nn.ReLU()
        )

        self.hidden2 = nn.Sequential(
            nn.Linear(in_features=12, out_features=8),
            nn.ReLU()
        )        
        
        self.output = nn.Sequential(
            nn.Linear(8, n_outputs_1),
        )

        self.NIG = nn.Sequential(
            nn.Linear(8, 4*self.n_actions*2*self.n_quantiles), # = 4 * 2 * n_outputs_1
        )

        self.hidden1.apply(lambda x: init_weights(x, self.weight_scale))
        self.hidden2.apply(lambda x: init_weights(x, self.weight_scale))
        self.output.apply(lambda x: init_weights(x, self.weight_scale))
        self.NIG.apply(lambda x: init_weights(x, self.weight_scale))

    def evi_split(self, out):
        mu, logv, logalpha, logbeta = torch.split(out, self.n_actions*2*self.n_quantiles, dim=-1)

        v = F.softplus(logv)
        alpha = F.softplus(logalpha) + 1
        beta = F.softplus(logbeta)
        return torch.concat([mu, v, alpha, beta], axis=-1)

    def forward(self, obs):
        obs = self.hidden1(obs)
        hidden = self.hidden2(obs)
        output = self.output(hidden)
        G_evi = self.NIG(hidden)
        G_evi = self.evi_split(G_evi)

        return output, G_evi