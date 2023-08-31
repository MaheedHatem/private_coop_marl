from config import Config
import numpy as np

class BaseController:
    def __init__(self, n_agents, names, obs_dim, act_dim,
            config, rng = None):
        self.n_agents = n_agents
        self.names = names
        self.use_full_obs = config.use_full_obs
        self.reward_sharing = config.reward_sharing

    def insert_experience(self, obs, act, 
        next_obs, rews, done, truncated, sample_id):
        raise NotImplementedError()

    def get_action(self, obs, deterministic = False):
        raise NotImplementedError()

    def train(self, number_of_batches, step):
        raise NotImplementedError()

    def update_epsilon(self, step):
        raise NotImplementedError()

    def save_models(self, save_dir, step):
        raise NotImplementedError()

    def load_models(self, save_dir, step):
        raise NotImplementedError()