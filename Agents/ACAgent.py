from .BaseAgent import *
from config import Config
from typing import Tuple, Callable, List
from misc import get_model, CategoricalModel
from copy import deepcopy
import torch
import torch.nn as nn

class ACAgent(BaseAgent):
    def __init__(self, name: str, obs_dim: Tuple, act_dim: int,
            config: Config, rng = None):
                super().__init__(name, obs_dim, act_dim, config, rng)
                self.critic = get_model(obs_dim, config.hidden_layers + [1], cnn=config.cnn).to(self.config.device)
                self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.config.lr)
                self.actor = CategoricalModel(obs_dim, config.hidden_layers + [act_dim])
                self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.config.lr)
                self.val_coef = config.value_coef
                self.entropy_coef = config.entropy_coef

    def get_action(self, obs: np.ndarray, determenistic=False) -> int:
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float32).to(self.config.device)
            dist = self.actor.get_distribution(torch.unsqueeze(obs, 0))
            return dist.sample().detach().numpy().item()
    
    def get_value(self, obs: np.ndarray) -> int:
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float32).to(self.config.device)
            return self.critic(torch.unsqueeze(obs, 0)).detach().numpy().item()


    def train_critic(self, critic: nn.Module, optimizer: torch.optim.Adam, batches: List[Tuple[np.ndarray]], use_other_ret: bool = False):
        advantages = []
        for batch in batches:
            batch = tuple(torch.as_tensor(data).to(self.config.device) for data in batch)
            obs, act, rewards, done, ret, other_ret = batch
            batch_idx = torch.arange(len(act)).long()
            optimizer.zero_grad()
            if use_other_ret:
                ret = other_ret
            adv = ret - critic(obs)
            loss = self.val_coef * ((adv)**2)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            advantages.append(adv.detach())
        return advantages

    def train_actor(self, actor: nn.Module, optimizer: torch.optim.Adam, batches: List[Tuple[np.ndarray]], advantages: List[torch.Tensor]):
        for batch, adv in zip(batches, advantages):
            batch = tuple(torch.as_tensor(data).to(self.config.device) for data in batch)
            obs, act, rewards, done, ret, _ = batch
            batch_idx = torch.arange(len(act)).long()
            optimizer.zero_grad()
            act_prob , entropy = actor.get_act_prob(obs, act)
            loss = -(adv*act_prob)
            loss = loss.mean() - self.entropy_coef * entropy
            loss.backward()
            optimizer.step()

    def train(self, number_of_batches: int, step: int):
        batches = self.replay.get_data()
        adv = self.train_critic(self.critic, self.critic_optimizer, batches)
        self.train_actor(self.actor, self.actor_optimizer, batches, adv)

    def save(self, save_dir: str, name: str, step: int):
        torch.save(self.critic.state_dict(), f"{save_dir}/{name}_critic{step}.pth")
        torch.save(self.actor.state_dict(), f"{save_dir}/{name}_actor{step}.pth")

    def load(self, save_dir: str, name: str, step: int):
        self.critic.load_state_dict(torch.load(f"{save_dir}/{name}_critic{step}.pth"))
        self.actor.load_state_dict(torch.load(f"{save_dir}/{name}_actor{step}.pth"))

    def finish_path(self, obs: np.ndarray, truncated: bool):
        last_value = 0.0
        if(truncated):
            last_value = self.get_value(obs)
        self.replay.finish_path(last_value)