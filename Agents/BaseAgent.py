from misc.ReplayBuffer import ReplayBuffer
from config import Config
from typing import Tuple, Dict
import numpy as np
import torch
import torch.nn as nn

class BaseAgent(nn.Module):
    def __init__(self, name: str, obs_dim: Tuple, act_dim: int,
            config: Config, rng = None):
        super().__init__()
        if rng == None:
            rng = np.random.default_rng()
        self.rng = rng
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.replay = ReplayBuffer(obs_dim, config.replay_size, 
            config.batch_size, config.gamma, rng)
        self.name = name
        self.config = config
        self.perturb_prob = config.perturb_prob

    def get_action(self, obs: np.ndarray, determenistic=False) -> int:
        raise  NotImplementedError()

    def insert_experience(self, obs: np.ndarray, act: np.ndarray, 
            next_obs: np.ndarray, reward: float, done: int, sample_id: int):
        self.replay.insert(obs, act, next_obs, reward, done, sample_id)
    
    def get_scores(self, trajectories: np.ndarray) -> np.ndarray:
        return self.replay.get_rewards(trajectories)
    
    def compare_trajectories(self, trajectories_a: np.ndarray, 
        trajectories_b: np.ndarray) -> np.ndarray:
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

    def train(self, number_of_batches: int, step: int):
        raise NotImplementedError()

    def update_epsilon(self, step: int):
        pass

    def save(self, save_dir: str, name: str, step: int):
        raise NotImplementedError()

    def load(self, save_dir: str, name: str, step: int):
        raise NotImplementedError()

    def finish_path(self, obs: Dict[str, np.ndarray], truncated: bool):
        pass