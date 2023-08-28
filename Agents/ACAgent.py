from .BaseAgent import *
from config import Config
from typing import Tuple, Callable, List
from misc import get_model, CategoricalModel
from copy import deepcopy
from itertools import chain
import torch
import torch.nn as nn

class ACAgent(BaseAgent):
    def __init__(self, name: str, obs_dim: Tuple, act_dim: int,
            config: Config, rng = None):
                super().__init__(name, obs_dim, act_dim, config, rng)
                self.critic = get_model(obs_dim, config.hidden_layers + [1], cnn=config.cnn).to(self.config.device)
                self.actor = CategoricalModel(obs_dim, config.hidden_layers + [act_dim])
                self.optimizer = torch.optim.Adam(chain(self.critic.parameters(), self.actor.parameters()), self.config.lr, eps=config.adam_eps)
                self.val_coef = config.value_coef
                self.entropy_coef = config.entropy_coef

    def get_action(self, obs: np.ndarray, determenistic=False) -> int:
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float32).to(self.config.device)
            dist = self.actor.get_distribution(obs)
            return dist.sample().detach().numpy()
    
    def get_value(self, obs: np.ndarray) -> int:
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float32).to(self.config.device)
            return self.critic(obs).detach().numpy()

    def insert_experience(self, obs: np.ndarray, act: np.ndarray, 
            next_obs: np.ndarray, reward: float, done: int, truncated: bool, sample_id: int):
        val = self.get_value(next_obs)
        if isinstance(val, tuple):
            other_val = val[1]
            self.replay.insert_other_val(other_val)
            val = val[0]
        self.replay.insert_val(val)
        super().insert_experience(obs, act, next_obs, reward, done, truncated, sample_id)

    def normalize(self, adv: torch.Tensor) -> torch.Tensor:
        std = torch.std(adv)
        if std == .0:
            std = 1
        return (adv - torch.mean(adv))/std

    def critic_loss(self, critic: nn.Module, batches: List[Tuple[np.ndarray]], use_other_ret: bool = False):
        advantages = []
        loss = 0
        for batch in batches:
            batch = tuple(torch.as_tensor(data).to(self.config.device) for data in batch)
            obs, act, rewards, ret, other_ret = batch
            batch_idx = torch.arange(len(act)).long()
            if use_other_ret:
                ret = other_ret
            adv = ret - critic(obs)
            loss += ((adv)**2).mean()
            advantages.append(adv.detach())
        return loss, advantages

    def actor_loss(self, actor: nn.Module, batches: List[Tuple[np.ndarray]], advantages: List[torch.Tensor]):
        loss = 0
        for batch, adv in zip(batches, advantages):
            batch = tuple(torch.as_tensor(data).to(self.config.device) for data in batch)
            obs, act, rewards, ret, _ = batch
            batch_idx = torch.arange(len(act)).long()
            act_prob, entropy = actor.get_act_prob(obs, act)
            loss += -(adv*act_prob).mean() - self.entropy_coef * entropy
        return loss

    def train(self, number_of_batches: int, step: int):
        batches = self.replay.get_data()
        self.optimizer.zero_grad()
        critic_loss, adv = self.train_critic(self.critic, batches)
        adv = [self.normalize(a) for a in adv]
        actor_loss = self.train_actor(self.actor, batches, adv)
        loss = self.val_coef * critic_loss + actor_loss
        loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optimizer.step()

    def save(self, save_dir: str, name: str, step: int):
        torch.save(self.critic.state_dict(), f"{save_dir}/{name}_critic{step}.pth")
        torch.save(self.actor.state_dict(), f"{save_dir}/{name}_actor{step}.pth")

    def load(self, save_dir: str, name: str, step: int):
        self.critic.load_state_dict(torch.load(f"{save_dir}/{name}_critic{step}.pth"))
        self.actor.load_state_dict(torch.load(f"{save_dir}/{name}_actor{step}.pth"))