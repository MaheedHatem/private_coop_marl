from .BaseAgent import *
from config import Config
from misc import get_model, CategoricalModel
from copy import deepcopy
from itertools import chain
import torch
import torch.nn as nn
from .ACAgent import ACAgent
from .RewardPredictor import RewardPredictor

class ACRewardAgent(ACAgent):
    def __init__(self, name, obs_dim, act_dim,
            config, rng = None):
                super().__init__(name, obs_dim, act_dim, config, rng)
                self.other_critic = get_model(obs_dim, config.hidden_layers + [1], cnn=config.cnn).to(self.config.device)
                self.optimizer = torch.optim.Adam(chain(self.critic.parameters(), self.actor.parameters(), self.other_critic.parameters()), self.config.lr, eps=config.adam_eps)
                self.predictor = RewardPredictor(obs_dim, act_dim, config)
                self.reward_weighting = config.reward_weighting
                self.other_value_coef = config.other_value_coef

    def get_action(self, obs, determenistic=False):
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float32).to(self.config.device)
            dist = self.actor.get_distribution(obs)
            return dist.sample().detach().cpu().numpy()
    
    def get_value(self, obs):
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float32).to(self.config.device)
            return self.critic(obs).detach().numpy(), self.other_critic(obs).detach().numpy()

    def train(self, number_of_batches, step):
        data = self.replay.get_data()
        data = tuple(torch.as_tensor(d).to(self.config.device) for d in data)
        self.optimizer.zero_grad()
        critic_loss, adv = self.critic_loss(self.critic, data)
        other_critic_loss, other_adv = self.critic_loss(self.other_critic, data, use_other_ret=True)
        adv = self.normalize(adv)
        other_adv = self.normalize(other_adv)
        #combined_adv = self.reward_weighting * adv + (1-self.reward_weighting) * other_adv
        actor_loss, entropy = self.actor_loss(self.actor, data, adv)
        other_actor_loss, _ = self.actor_loss(self.actor, data, other_adv)
        loss = self.val_coef * critic_loss + actor_loss + self.reward_weighting * other_actor_loss + \
                 self.other_value_coef * other_critic_loss  - self.entropy_coef * entropy
        loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.other_critic.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optimizer.step()

    def save(self, save_dir, name, step):
        torch.save(self.critic.state_dict(), f"{save_dir}/{name}_critic{step}.pth")
        torch.save(self.actor.state_dict(), f"{save_dir}/{name}_actor{step}.pth")
        torch.save(self.other_critic.state_dict(), f"{save_dir}/{name}_other_critic{step}.pth")
        self.predictor.save(save_dir, name, step)

    def load(self, save_dir, name, step):
        self.critic.load_state_dict(torch.load(f"{save_dir}/{name}_critic{step}.pth"))
        self.actor.load_state_dict(torch.load(f"{save_dir}/{name}_actor{step}.pth"))
        self.other_critic.load_state_dict(torch.load(f"{save_dir}/{name}_other_critic{step}.pth"))
        self.predictor.load(save_dir, name, step)

    def insert_experience(self, obs, act, 
            next_obs, reward, done, truncated, sample_id):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).to(self.config.device)
        act_tensor = torch.as_tensor(act, dtype=torch.float32).to(self.config.device)
        self.replay.insert_other_reward(self.predictor.get_reward(obs_tensor, act_tensor))
        super().insert_experience(obs, act, next_obs, reward, done, truncated, sample_id)

    def get_scores(self, trajectories):
        scores = super().get_scores(trajectories)
        trajectories = torch.as_tensor(self.replay.get_obs(trajectories)).to(self.config.device)
        with torch.no_grad():
            trajectories = trajectories.index_select(1, torch.tensor([trajectories.shape[1]-1]).to(self.config.device))
            scores = self.critic(trajectories).squeeze() + scores
            return scores

    def train_predictor(self, trajectory_a, trajectory_b, ratio):
        self.predictor.train(self.replay.get_obs(trajectory_a), self.replay.get_obs(trajectory_b), 
            self.replay.get_act(trajectory_a), self.replay.get_act(trajectory_b), ratio)