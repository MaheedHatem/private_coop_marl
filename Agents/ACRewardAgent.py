from .BaseAgent import *
from config import Config
from typing import Tuple, Callable, List
from misc import get_model, CategoricalModel
from copy import deepcopy
import torch
import torch.nn as nn
from .ACAgent import ACAgent
from .RewardPredictor import RewardPredictor

class ACRewardAgent(ACAgent):
    def __init__(self, name: str, obs_dim: Tuple, act_dim: int,
            config: Config, rng = None):
                super().__init__(name, obs_dim, act_dim, config, rng)
                self.other_critic = get_model(obs_dim, config.hidden_layers + [1], cnn=config.cnn).to(self.config.device)
                self.other_critic_optimizer = torch.optim.Adam(self.other_critic.parameters(), self.config.lr)
                self.other_actor = CategoricalModel(obs_dim, config.hidden_layers + [act_dim])
                self.other_actor_optimizer = torch.optim.Adam(self.other_actor.parameters(), self.config.lr)
                self.predictor = RewardPredictor(obs_dim, act_dim, config)
                self.reward_weighting = config.reward_weighting

    def get_action(self, obs: np.ndarray, determenistic=False) -> int:
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float32).to(self.config.device)
            if self.rng.random() < self.reward_weighting:
                dist = self.actor.get_distribution(torch.unsqueeze(obs, 0))
                return dist.sample().detach().numpy().item()
            else:
                dist = self.other_actor.get_distribution(torch.unsqueeze(obs, 0))
                return dist.sample().detach().numpy().item()
    
    def get_value(self, obs: np.ndarray) -> int:
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float32).to(self.config.device)
            return self.critic(torch.unsqueeze(obs, 0)).detach().numpy().item(), self.other_critic(torch.unsqueeze(obs, 0)).detach().numpy().item()

    def train(self, number_of_batches: int, step: int):
        batches = self.replay.get_data()
        adv = self.train_critic(self.critic, self.critic_optimizer, batches)
        other_adv = self.train_critic(self.other_critic, self.other_critic_optimizer, batches, True)
        #combined_adv = [self.reward_weighting * a + (1 - self.reward_weighting) * oa for a, oa in zip(adv, other_adv)]
        self.train_actor(self.actor, self.actor_optimizer, batches, adv)
        self.train_actor(self.other_actor, self.other_actor_optimizer, batches, other_adv)

    def save(self, save_dir: str, name: str, step: int):
        torch.save(self.critic.state_dict(), f"{save_dir}/{name}_critic{step}.pth")
        torch.save(self.actor.state_dict(), f"{save_dir}/{name}_actor{step}.pth")
        torch.save(self.other_critic.state_dict(), f"{save_dir}/{name}_other_critic{step}.pth")
        torch.save(self.other_actor.state_dict(), f"{save_dir}/{name}_other_actor{step}.pth")
        self.predictor.save(save_dir, name, step)

    def load(self, save_dir: str, name: str, step: int):
        self.critic.load_state_dict(torch.load(f"{save_dir}/{name}_critic{step}.pth"))
        self.actor.load_state_dict(torch.load(f"{save_dir}/{name}_actor{step}.pth"))
        self.other_critic.load_state_dict(torch.load(f"{save_dir}/{name}_other_critic{step}.pth"))
        self.other_actor.load_state_dict(torch.load(f"{save_dir}/{name}_other_actor{step}.pth"))
        self.predictor.load(save_dir, name, step)

    def insert_experience(self, obs: np.ndarray, act: np.ndarray, 
            next_obs: np.ndarray, reward: float, done: int, sample_id: int):
        obs = torch.unsqueeze(torch.as_tensor(obs, dtype=torch.float32).to(self.config.device), 0)
        act = torch.unsqueeze(torch.as_tensor(act, dtype=torch.float32).to(self.config.device), 0)
        self.replay.insert_other_reward(self.predictor.get_reward(obs, act))
        super().insert_experience(obs, act, next_obs, reward, done, sample_id)

    def get_scores(self, trajectories: np.ndarray) -> np.ndarray:
        trajectories = torch.as_tensor(self.replay.get_obs(trajectories)).to(self.config.device)
        with torch.no_grad():
            scores = self.critic(trajectories.view(-1, *trajectories.shape[2:]))
            scores = scores.view(trajectories.shape[0], trajectories.shape[1])
            scores = scores.sum(dim=-1)
            return scores

    def train_predictor(self, trajectory_a: np.ndarray, trajectory_b: np.ndarray, ratio: np.ndarray):
        self.predictor.train(self.replay.get_obs(trajectory_a), self.replay.get_obs(trajectory_b), 
            self.replay.get_act(trajectory_a), self.replay.get_act(trajectory_b), ratio)

    def finish_path(self, obs: np.ndarray, truncated: bool):
        last_value = (0.0, 0.0)
        if(truncated):
            last_value = self.get_value(obs)
        self.replay.finish_path(last_value)