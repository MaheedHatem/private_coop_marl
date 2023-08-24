from .BaseAgent import *
from config import Config
from typing import Tuple, Callable
from misc import get_model
from copy import deepcopy
import torch
import torch.nn as nn

class DQNAgent(BaseAgent):
    def __init__(self, name: str, obs_dim: Tuple, act_dim: int,
            config: Config, rng = None):
                super().__init__(name, obs_dim, act_dim, config, rng)
                self.q_network = get_model(obs_dim, config.hidden_layers + [act_dim], cnn=config.cnn).to(self.config.device)
                self.target_network = deepcopy(self.q_network).to(self.config.device).to(self.config.device)
                self.optimizer = torch.optim.Adam(self.q_network.parameters(), self.config.lr, eps=config.adam_eps)
                self.epsilon = config.init_epsilon
                self.target_update_every = config.target_update_every

    def get_action(self, obs: np.ndarray, determenistic=False) -> int:
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float32).to(self.config.device)
            random_number = self.rng.random()
            if random_number > self.epsilon or determenistic:
                q_val = self.q_network(obs, 0)
                return torch.argmax(q_val, axis=-1).cpu().detach().numpy()
            else:
                return self.rng.integers(self.act_dim, size=(self.num_parallel, 1))

    def get_targets(self, target_net: nn.Module, obs: torch.Tensor, act: torch.Tensor, next_obs: torch.Tensor, rewards: torch.Tensor, done: torch.Tensor):
        with torch.no_grad():
            return rewards + (1 - done) * self.config.gamma * target_net(next_obs).max(dim=-1).values


    def train_q_network(self, number_of_batches: int, q_net: nn.Module, target_net: nn.Module,
         get_targets: Callable, optimizer: torch.optim.Adam, update_target: bool = False):
        for b in range(number_of_batches):
            batch = self.replay.get_batch()
            batch = tuple(torch.as_tensor(data).to(self.config.device) for data in batch)
            obs, act, next_obs, rewards, done = batch
            batch_idx = torch.arange(len(act)).long()
            target_q = get_targets(target_net, obs, act, next_obs, rewards, done)
            optimizer.zero_grad()
            loss = ((q_net(obs)[batch_idx, act]  - target_q)**2)
            loss = loss.mean()
            loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), self.max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                for p, p_targ in zip(q_net.parameters(), target_net.parameters()):
                    p_targ.data.copy_((1-self.config.target_update)* p_targ.data + 
                        self.config.target_update * p.data)

    
    def update_epsilon(self, step: int):
        self.epsilon = max(self.config.final_epsilon, self.config.init_epsilon + step * (self.config.final_epsilon - self.config.init_epsilon)/self.config.random_steps)

    def train(self, number_of_batches: int, step: int):
        self.train_q_network(number_of_batches, self.q_network, self.target_network, self.get_targets, self.optimizer, step%self.target_update_every==0)

    def save(self, save_dir: str, name: str, step: int):
        torch.save(self.q_network.state_dict(), f"{save_dir}/{name}_q{step}.pth")

    def load(self, save_dir: str, name: str, step: int):
        self.q_network.load_state_dict(torch.load(f"{save_dir}/{name}_q{step}.pth"))