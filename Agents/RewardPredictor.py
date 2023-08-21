import torch
import torch.nn as nn
from config import Config
from itertools import chain
from misc import get_model
import numpy as np
import torch.nn as nn

class RewardPredictor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, config: Config):
        super().__init__()
        self.device = config.device
        self.act_dim = act_dim
        self.models = [get_model(np.product(obs_dim)+self.act_dim, config.reward_hidden_layers + [1], cnn=config.cnn, output_activation=nn.Tanh()).to(self.device) for _ in range(config.N_predictors)]
        #self.models = [get_model(np.product(obs_dim)+1, config.reward_hidden_layers + [1], cnn=config.cnn).to(self.device) for _ in range(config.N_predictors)]
        self.optimizer = torch.optim.Adam(chain(*[model.parameters() for model in self.models]), lr=config.reward_lr)
        self.max_grad_norm = config.max_grad_norm

    def get_act_one_hot(self, act: torch.Tensor) -> torch.Tensor:

        act_one_hot = torch.zeros((*act.shape, self.act_dim)).scatter_(-1, act.long().unsqueeze(-1), 1).to(self.device)

        return act_one_hot
        #return torch.unsqueeze(act, -1)
    
    def get_reward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            rewards = 0
            act = self.get_act_one_hot(act)
            obs_act = torch.cat([obs, act], dim=-1)
            for model in self.models:
                rewards += model(obs_act)
            rewards /= len(self.models)
        return rewards

    def train(self, trajectory_a: np.ndarray, trajectory_b: np.ndarray, 
            act_a: np.ndarray, act_b: np.ndarray, ratio: np.ndarray): 
        #trajectory dim = Number trajectories X obs per trajectories X obs dim
        #ratio is 1 if a is preferred
        act_a = torch.as_tensor(act_a, dtype = torch.float32).to(self.device)
        act_b = torch.as_tensor(act_b, dtype = torch.float32).to(self.device)
        trajectory_a = torch.as_tensor(trajectory_a, dtype = torch.float32).to(self.device)
        trajectory_b = torch.as_tensor(trajectory_b, dtype = torch.float32).to(self.device)
        act_a = self.get_act_one_hot(act_a)
        act_b = self.get_act_one_hot(act_b)
        trajectory_a = torch.cat([trajectory_a, act_a], axis=-1)
        trajectory_b = torch.cat([trajectory_b, act_b], axis=-1)
        ratio = torch.as_tensor(ratio).to(self.device)
        self.optimizer.zero_grad()
        loss = 0
        for model in self.models:
            scores_a = torch.squeeze(model(trajectory_a.view(-1, *trajectory_a.shape[2:])))
            scores_b = torch.squeeze(model(trajectory_b.view(-1, *trajectory_b.shape[2:])))
            scores_a = scores_a.view(trajectory_a.shape[0], trajectory_a.shape[1])
            scores_b = scores_b.view(trajectory_b.shape[0], trajectory_b.shape[1])
            scores_a = torch.exp(scores_a.sum(dim=-1))
            scores_b = torch.exp(scores_b.sum(dim=-1))
            total_scores = scores_a + scores_b
            p_a = scores_a/total_scores
            p_b = scores_b/total_scores
            log_p_a = torch.log(p_a)
            log_p_b = torch.log(p_b)
            assert(scores_a.shape[0] == trajectory_a.shape[0])
            assert(len(scores_a.shape) == 1)
            loss -= (ratio * log_p_a + (1-ratio) * log_p_b).mean()
            if(torch.isnan(loss)):
                raise Exception("loss is nan")

        loss.backward()
        for model in self.models:
            nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
        self.optimizer.step()

    def save(self, save_dir: str, name: str, step: int):
        for i in range(len(self.models)):
            torch.save(self.models[i].state_dict(), f"{save_dir}/{name}_pred_{i}_{step}.pth")

    def load(self, save_dir: str, name: str, step: int):
        for i in range(len(self.models)):
            self.models[i].load_state_dict(torch.load(f"{save_dir}/{name}_pred_{i}_{step}.pth"))