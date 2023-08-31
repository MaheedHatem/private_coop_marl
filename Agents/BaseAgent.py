from misc.ReplayBuffer import ReplayBuffer
from config import Config
import numpy as np
import torch
import torch.nn as nn

class BaseAgent(nn.Module):
    def __init__(self, name, obs_dim, act_dim,
            config, rng = None):
        super().__init__()
        if rng == None:
            rng = np.random.default_rng()
        self.rng = rng
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_parallel = config.num_parallel
        self.replay = ReplayBuffer(obs_dim, config.num_parallel, config.replay_size, config.trajectory_database,
            config.batch_size, config.gamma, rng)
        self.max_grad_norm = config.max_grad_norm
        self.name = name
        self.config = config
        if config.reward_sharing:
            self.perturb_prob = config.perturb_prob

    def get_action(self, obs, determenistic=False):
        raise  NotImplementedError()

    def insert_experience(self, obs, act, 
            next_obs, reward, done, truncated, sample_id):
        self.replay.insert(obs, act, next_obs, reward, done, truncated, sample_id)
    
    def get_scores(self, trajectories):
        return self.replay.get_rewards(trajectories)
    
    def compare_trajectories(self, trajectories_a, 
        trajectories_b):
        scores_a = self.get_scores(trajectories_a)
        scores_b = self.get_scores(trajectories_b)
        preference = np.zeros(len(scores_a))
        diff = scores_b - scores_a
        preference[np.abs(diff) <= self.config.similarity] = 0.5
        indx_b_better = diff > self.config.similarity
        indx_a_better = (-diff) > self.config.similarity
        preference[indx_b_better] = 0
        preference[indx_a_better] = 1
        # preference[indx_b_better] = np.exp(scores_a[indx_b_better]) / np.exp(scores_b[indx_b_better])
        # preference[indx_a_better] = 1 - np.exp(scores_b[indx_a_better]) / np.exp(scores_a[indx_a_better])
        random_choices = np.array([0, 1, 0.5])
        for i in range(len(preference)):
            if self.rng.random() < self.perturb_prob:
                preference[i] = random_choices[self.rng.integers(len(random_choices))]
                #preference[i] = self.rng.random()
        return preference

    def train(self, number_of_batches, step):
        raise NotImplementedError()

    def update_epsilon(self, step):
        pass

    def save(self, save_dir, name, step):
        raise NotImplementedError()

    def load(self, save_dir, name, step):
        raise NotImplementedError()
    