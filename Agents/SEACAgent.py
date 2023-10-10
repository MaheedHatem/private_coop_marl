from .BaseAgent import *
from .ACAgent import *
from config import Config
from misc import get_model, CategoricalModel
from copy import deepcopy
from itertools import chain
import torch
import torch.nn as nn

class SEACAgent(ACAgent):
    def __init__(self, name, obs_dim, act_dim,
            config, rng = None):
                super().__init__(name, obs_dim, act_dim, config, rng)
                self.other_buffers = None

    def init_buffer_list(self, other_buffers):
        assert self.other_buffers == None
        self.other_buffers = other_buffers
    
    def get_logp(self, obs, act):
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float32).to(self.config.device)
            act = torch.as_tensor(act, dtype=torch.float32).to(self.config.device)
            act_prob, _ = self.actor.get_act_prob(obs, act)
            return act_prob

    def insert_experience(self, obs, act, 
            next_obs, reward, done, truncated, sample_id):
        act_prob = self.get_logp(obs, act)
        super().insert_experience(obs, act, next_obs, reward, done, truncated, sample_id)

    def seac_loss(self, critic, actor, data):
        obs, act, rewards, ret, old_logp = data
        adv = ret - critic(obs)
        act_prob, _ = actor.get_act_prob(obs, act)
        importance_sampling = (act_prob.exp() / (old_logp.exp() + 1e-7)).detach()
        seac_value_loss = (importance_sampling * (adv**2)).mean()
        #seac_policy_loss = (-importance_sampling * act_prob * adv.detach()).mean()
        seac_policy_loss = (-importance_sampling * act_prob * self.normalize(adv.detach())).mean()
        return seac_value_loss, seac_policy_loss


    def train(self, number_of_batches, step):
        data = self.replay.get_data()
        data = tuple(torch.as_tensor(d).to(self.config.device) for d in data)
        self.optimizer.zero_grad()
        critic_loss, adv = self.critic_loss(self.critic, data)
        adv = self.normalize(adv)
        actor_loss, entropy = self.actor_loss(self.actor, data, adv)
        seac_value_loss = 0
        seac_policy_loss = 0
        for buffer in self.other_buffers:
            data = buffer.get_seac_data()
            data = tuple(torch.as_tensor(d).to(self.config.device) for d in data)
            current_seac_value_loss, current_seac_policy_loss = self.seac_loss(self.critic, self.actor, data)
            seac_value_loss += current_seac_value_loss
            seac_policy_loss += current_seac_policy_loss
        loss = self.val_coef * critic_loss + actor_loss - self.entropy_coef * entropy + seac_policy_loss + self.val_coef * seac_value_loss
        loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optimizer.step()

    def get_replay(self):
        return self.replay
