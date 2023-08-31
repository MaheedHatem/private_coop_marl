from .DQNAgent import DQNAgent
from .RewardPredictor import RewardPredictor
from misc import get_model
import numpy as np
from config import Config
from copy import deepcopy
import torch
import torch.nn as nn

class DQNRewardAgent(DQNAgent):
    def __init__(self, name, obs_dim, act_dim,
            config, rng = None):
                super().__init__(name, obs_dim, act_dim, config, rng)
                self.predictor = RewardPredictor(obs_dim, act_dim, config)
                self.other_q = get_model(obs_dim, config.hidden_layers + [act_dim], cnn=config.cnn).to(self.config.device)
                self.other_target = deepcopy(self.other_q).to(self.config.device).to(self.config.device)
                self.other_optimizer = torch.optim.Adam(self.other_q.parameters(), self.config.lr, eps=config.adam_eps)
                self.reward_weighting = config.reward_weighting

    def get_q_predict(self, obs):
        obs = obs.repeat((self.act_dim,1))
        act = torch.arange(self.act_dim)
        return self.predictor.get_reward(obs, act)
        
    def get_action(self, obs, determenistic=False):
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float32).to(self.config.device)
            random_number = self.rng.random()
            if random_number > self.epsilon or determenistic:
                q_val = nn.functional.normalize(self.q_network(obs), p=1, dim=1)
                other_q_val = nn.functional.normalize(self.other_q(obs), p=1, dim=1)
                #other_q_val = nn.functional.normalize(self.get_q_predict(obs), p=1, dim=1)
                act_scores = self.reward_weighting * q_val + (1-self.reward_weighting) * other_q_val
                if self.rng.random() < self.reward_weighting:
                    return torch.argmax(q_val, axis=-1).cpu().detach().numpy()
                else:
                    return torch.argmax(other_q_val, axis=-1).cpu().detach().numpy()
                return torch.argmax(act_scores, axis=-1).cpu().detach().numpy()
            else:
                return self.rng.integers(self.act_dim, size=(self.num_parallel, 1))

    def get_targets_predicted(self, target_net, obs, act, next_obs, 
        rewards, done):
        with torch.no_grad():
            predicted_reward = torch.squeeze(self.predictor.get_reward(obs, act))
            return predicted_reward + (1 - done) * self.config.gamma * target_net(next_obs).max(dim=-1).values

    def get_scores(self, trajectories):
        scores = super().get_scores(trajectories)
        trajectories = torch.as_tensor(self.replay.get_obs(trajectories)).to(self.config.device)
        with torch.no_grad():
            trajectories = trajectories.index_select(1, torch.tensor([trajectories.shape[1]-1]))
            scores = self.target_network(trajectories).max(dim=-1).values + scores
            return scores

    def train_predictor(self, trajectory_a, trajectory_b, ratio):
        self.predictor.train(self.replay.get_obs(trajectory_a), self.replay.get_obs(trajectory_b), 
            self.replay.get_act(trajectory_a), self.replay.get_act(trajectory_b), ratio)

    def train(self, number_of_batches, step):
        self.train_q_network(number_of_batches, self.q_network, self.target_network, self.get_targets, self.optimizer,  step%self.target_update_every==0)
        self.train_q_network(number_of_batches, self.other_q, self.other_target, self.get_targets_predicted, self.other_optimizer, step%self.target_update_every==0)

    def save(self, save_dir, name, step):
        torch.save(self.q_network.state_dict(), f"{save_dir}/{name}_q{step}.pth")
        torch.save(self.other_q.state_dict(), f"{save_dir}/{name}_oq{step}.pth")
        self.predictor.save(save_dir, name, step)

    def load(self, save_dir, name, step):
        self.q_network.load_state_dict(torch.load(f"{save_dir}/{name}_q{step}.pth"))
        self.other_q.load_state_dict(torch.load(f"{save_dir}/{name}_oq{step}.pth"))
        self.predictor.load(save_dir, name, step)