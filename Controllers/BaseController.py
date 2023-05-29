from typing import List, Dict, Tuple
from config import Config
import numpy as np

class BaseController:
    def __init__(self, n_agents: int, names: List[str], obs_dim: Dict[str, Tuple], act_dim: Dict[str, int],
            config: Config, rng = None):
        self.n_agents = n_agents
        self.names = names
        self.use_full_obs = config.use_full_obs
        self.reward_sharing = config.reward_sharing

    def insert_experience(self, obs: Dict[str, np.ndarray], act: Dict[str, np.ndarray], 
        next_obs: Dict[str, np.ndarray], rews: Dict[str, np.ndarray], done :[str, np.ndarray], sample_id: int):
        raise NotImplementedError()

    def get_action(self, obs: Dict[str, np.ndarray], deterministic: bool = False) -> Dict[str, np.ndarray]:
        raise NotImplementedError()

    def train(self, number_of_batches: int, train_pred: bool = True, update_target: bool = False):
        raise NotImplementedError()

    def update_epsilon(self, step: int):
        raise NotImplementedError()

    def save_models(self, save_dir: str, step: int):
        raise NotImplementedError()

    def load_models(self, save_dir: str, step: int):
        raise NotImplementedError()