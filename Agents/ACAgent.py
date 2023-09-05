from .BaseAgent import *
from config import Config
from misc import get_model, CategoricalModel
from copy import deepcopy
from itertools import chain
import torch
import torch.nn as nn

class ACAgent(BaseAgent):
    def __init__(self, name, obs_dim, act_dim,
            config, rng = None):
                super().__init__(name, obs_dim, act_dim, config, rng)
                self.critic = get_model(obs_dim, config.hidden_layers + [1], cnn=config.cnn).to(self.config.device)
                self.actor = CategoricalModel(obs_dim, config.hidden_layers + [act_dim]).to(self.config.device)
                self.optimizer = torch.optim.Adam(chain(self.critic.parameters(), self.actor.parameters()), self.config.lr, eps=config.adam_eps)
                self.val_coef = config.value_coef
                self.entropy_coef = config.entropy_coef

    def get_action(self, obs, determenistic=False):
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float32).to(self.config.device)
            dist = self.actor.get_distribution(obs)
            return dist.sample().detach().cpu().numpy()
    
    def get_value(self, obs):
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float32).to(self.config.device)
            return self.critic(obs).detach().cpu().numpy()

    def insert_experience(self, obs, act, 
            next_obs, reward, done, truncated, sample_id):
        val = self.get_value(next_obs)
        if isinstance(val, tuple):
            other_val = val[1]
            self.replay.insert_other_val(other_val)
            val = val[0]
        self.replay.insert_val(val)
        super().insert_experience(obs, act, next_obs, reward, done, truncated, sample_id)

    def normalize(self, adv):
        std = torch.std(adv)
        if std == .0:
            std = 1
        return (adv - torch.mean(adv))/std

    def critic_loss(self, critic, data, use_other_ret = False):
        obs, act, rewards, ret, other_ret = data
        if use_other_ret:
            ret = other_ret
        adv = ret - critic(obs)
        loss = ((adv)**2).mean()
        return loss, adv.detach()

    def actor_loss(self, actor, data, advantages):
        
        obs, act, _, _, _ = data
        batch_idx = torch.arange(len(act)).long()
        act_prob, entropy = actor.get_act_prob(obs, act)
        loss = -(advantages*act_prob).mean()
        return loss, entropy

    def train(self, number_of_batches, step):
        data = self.replay.get_data()
        data = tuple(torch.as_tensor(d).to(self.config.device) for d in data)
        self.optimizer.zero_grad()
        critic_loss, adv = self.train_critic(self.critic, data)
        adv = self.normalize(adv)
        actor_loss, entropy = self.train_actor(self.actor, data, adv)
        loss = self.val_coef * critic_loss + actor_loss - self.entropy_coef * entropy
        loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optimizer.step()

    def save(self, save_dir, name, step):
        torch.save(self.critic.state_dict(), f"{save_dir}/{name}_critic{step}.pth")
        torch.save(self.actor.state_dict(), f"{save_dir}/{name}_actor{step}.pth")

    def load(self, save_dir, name, step):
        self.critic.load_state_dict(torch.load(f"{save_dir}/{name}_critic{step}.pth"))
        self.actor.load_state_dict(torch.load(f"{save_dir}/{name}_actor{step}.pth"))