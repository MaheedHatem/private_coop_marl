from typing import List, Dict, Tuple
from config import Config
from .BaseController import BaseController
from Agents import get_agent
import numpy as np

class CentralizedController(BaseController):
    def __init__(self, n_agents: int, names: List[str], obs_dim: Dict[str, Tuple], act_dim: Dict[str, int],
            config: Config, rng = None):
        super().__init__(n_agents, names, obs_dim, act_dim, config, rng)
        self.act_dim = [act_dim[name] for name in self.names]
        combined_act_dim = 1
        combined_obs_dim = list(obs_dim[self.names[0]])
        combined_obs_dim[0] *= self.n_agents
        combined_obs_dim = tuple(combined_obs_dim)
        for name in self.names:
            combined_act_dim *= act_dim[name]
        agent_init = get_agent(config.agent)
        self.agents = agent_init(name, combined_obs_dim, combined_act_dim, config, rng)

    def insert_experience(self, obs: Dict[str, np.ndarray], act: Dict[str, np.ndarray], 
        next_obs: Dict[str, np.ndarray], rews: Dict[str, np.ndarray], done :Dict[str, np.ndarray], truncated : bool, sample_id: int):
        obs = np.concatenate([obs[name] for name in self.names], axis=-1)
        next_obs = np.concatenate([next_obs[name] for name in self.names], axis=-1)
        act = self.centralized_action(act)
        rews = np.sum(np.array([rews[name] for name in self.names]), axis=0)
        self.agents.insert_experience(obs, act, next_obs, rews, done, truncated, sample_id)

    def decentralized_action(self, act: np.ndarray) -> Dict[str, int]:
        acts = np.zeros((len(act), self.n_agents), dtype=np.int64)
        for i in range(len(act)):
            a = act[i]
            index = 0
            while(a):
                acts[i, index] = a % self.act_dim[index]
                a //= self.act_dim[index]
                index += 1
        return acts

    def centralized_action(self, acts: Dict[str, int]):
        act = np.zeros(acts.shape[1], dtype=np.int64)
        for i,name in reversed(list(enumerate(self.names))):
            act *= self.act_dim[i]
            act += acts[name]
        return act

    def get_action(self, obs: Dict[str, np.ndarray], deterministic: bool = False) -> Dict[str, np.ndarray]:
        return self.decentralized_action(self.agents.get_action(np.concatenate([obs[name] for name in self.names], axis = -1), deterministic))
    
    def train(self, number_of_batches: int, step: int):
        self.agents.train(number_of_batches, step)

    def update_epsilon(self, step: int):
        self.agents.update_epsilon(step)

    def save_models(self, save_dir: str, step: int):
        self.agents.save(save_dir, 'central', step)

    def load_models(self, save_dir: str, step: int):
        self.agents.load(save_dir, 'central', step)