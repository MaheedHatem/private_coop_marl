from .BaseAgent import *
from config import Config
from typing import Tuple, Callable, List
from misc import get_model, CategoricalModel
from copy import deepcopy
from itertools import chain
import torch
import torch.nn as nn
from .ACAgent import ACAgent
from .RewardPredictor import RewardPredictor

class ACRewardAgent(ACAgent):
    def __init__(self, name: str, obs_dim: Tuple, act_dim: int,
            config: Config, rng = None):
                super().__init__(name, obs_dim, act_dim, config, rng)
                self.other_critic = get_model(obs_dim, config.hidden_layers + [1], cnn=config.cnn).to(self.config.device)
                self.optimizer = torch.optim.Adam(chain(self.critic.parameters(), self.actor.parameters(), self.other_critic.parameters()), self.config.lr, eps=config.adam_eps)
                self.predictor = RewardPredictor(obs_dim, act_dim, config)
                self.reward_weighting = config.reward_weighting

    def get_action(self, obs: np.ndarray, determenistic=False) -> int:
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float32).to(self.config.device)
            dist = self.actor.get_distribution(obs)
            return dist.sample().detach().numpy()
    
    def get_value(self, obs: np.ndarray) -> int:
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float32).to(self.config.device)
            return self.critic(obs).detach().numpy(), self.other_critic(obs).detach().numpy()

    def train(self, number_of_batches: int, step: int):
        batches = self.replay.get_data()
        self.optimizer.zero_grad()
        critic_loss, adv = self.critic_loss(self.critic, batches)
        other_critic_loss, other_adv = self.critic_loss(self.other_critic, batches, use_other_ret=True)
        adv = [self.normalize(a) for a in adv]
        other_adv = [self.normalize(a) for a in other_adv]
        combined_adv = [self.reward_weighting * a + (1-self.reward_weighting) * oa for a, oa in zip(adv, other_adv)]
        actor_loss = self.actor_loss(self.actor, batches, combined_adv)
        loss = self.val_coef * critic_loss + actor_loss + self.val_coef * other_critic_loss
        loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.other_critic.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optimizer.step()

    def save(self, save_dir: str, name: str, step: int):
        torch.save(self.critic.state_dict(), f"{save_dir}/{name}_critic{step}.pth")
        torch.save(self.actor.state_dict(), f"{save_dir}/{name}_actor{step}.pth")
        torch.save(self.other_critic.state_dict(), f"{save_dir}/{name}_other_critic{step}.pth")
        self.predictor.save(save_dir, name, step)

    def load(self, save_dir: str, name: str, step: int):
        self.critic.load_state_dict(torch.load(f"{save_dir}/{name}_critic{step}.pth"))
        self.actor.load_state_dict(torch.load(f"{save_dir}/{name}_actor{step}.pth"))
        self.other_critic.load_state_dict(torch.load(f"{save_dir}/{name}_other_critic{step}.pth"))
        self.predictor.load(save_dir, name, step)

    def insert_experience(self, obs: np.ndarray, act: np.ndarray, 
            next_obs: np.ndarray, reward: float, done: int, truncated: bool, sample_id: int):
        obs = torch.as_tensor(obs, dtype=torch.float32).to(self.config.device)
        act = torch.as_tensor(act, dtype=torch.float32).to(self.config.device)
        self.replay.insert_other_reward(self.predictor.get_reward(obs, act))
        super().insert_experience(obs, act, next_obs, reward, done, truncated, sample_id)

    def get_scores(self, trajectories: np.ndarray) -> np.ndarray:
        scores = super().get_scores(trajectories)
        trajectories = torch.as_tensor(self.replay.get_obs(trajectories)).to(self.config.device)
        with torch.no_grad():
            trajectories = trajectories.index_select(1, torch.tensor([trajectories.shape[1]-1]))
            scores = self.critic(trajectories).squeeze() + scores
            return scores

    def train_predictor(self, trajectory_a: np.ndarray, trajectory_b: np.ndarray, ratio: np.ndarray):
        self.predictor.train(self.replay.get_obs(trajectory_a), self.replay.get_obs(trajectory_b), 
            self.replay.get_act(trajectory_a), self.replay.get_act(trajectory_b), ratio)