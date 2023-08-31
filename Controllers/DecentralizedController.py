from config import Config
from .BaseController import BaseController
from .TrajectoryServer import TrajectoryServer
from Agents import get_agent
import numpy as np
import operator

class DecentralizedController(BaseController):
    def __init__(self, n_agents, names, obs_dim, act_dim,
            config, rng = None):
        super().__init__(n_agents, names, obs_dim, act_dim, config, rng)
        self.agents = {}
        agent_init = get_agent(config.agent)
        combined_obs_dim = list(obs_dim[self.names[0]])
        combined_obs_dim[0] *= self.n_agents
        combined_obs_dim = tuple(combined_obs_dim)
        for name in self.names:
            if self.use_full_obs:
                self.agents[name] = agent_init(name, combined_obs_dim, act_dim[name], config, rng)
            else:
                self.agents[name] = agent_init(name, obs_dim[name], act_dim[name], config, rng)
        
        if self.reward_sharing:
            self.trajectory_server = TrajectoryServer(config, self.agents, rng)
            self.reward_weighting = config.reward_weighting
        self.true_reward = config.true_reward
        self.update_predictor_steps = config.update_predictor_steps

    def insert_experience(self, obs, act, 
        next_obs, rews, done, truncated , sample_id):
        if self.use_full_obs:
            combined_obs = np.concatenate([obs[name] for name in self.names])
            combined_next_obs = np.concatenate([next_obs[name] for name in self.names])
        if self.reward_sharing:
            self.trajectory_server.insert_sample(sample_id)

        if self.true_reward:
            reward = np.sum(np.array([rews[name] for name in self.names]))
            rews = {name: reward for name in self.names}
        for name in self.names:
            if self.use_full_obs:
                self.agents[name].insert_experience(combined_obs, act[name], combined_next_obs, rews[name], done, truncated, sample_id)
            else:
                self.agents[name].insert_experience(obs[name], act[name], next_obs[name], rews[name], done, truncated, sample_id)

    def get_action(self, obs, deterministic= False):
        if self.use_full_obs:            
            combined_obs = np.concatenate([obs[name] for name in self.names])
        if self.use_full_obs:
            actions = np.column_stack([self.agents[name].get_action(combined_obs, deterministic) for name in self.names])
        else:
            actions = np.column_stack([self.agents[name].get_action(obs[name], deterministic) for name in self.names])
        return actions
    
    def train(self, number_of_batches, step):
        if self.reward_sharing and step < self.update_predictor_steps:
            for batch in range(number_of_batches):
                trajectories_a, trajectories_b, votes = self.trajectory_server.get_votes()
                if len(self.agents.values()) > 1:
                    for cur_name, agent in self.agents.items():
                        ratio = np.mean([votes[name] for name in self.names if name != cur_name], axis=0)
                        #ratio = np.mean([votes[name] for name in self.names], axis=0)
                        agent.train_predictor(trajectories_a, trajectories_b, ratio)
                else:
                    for cur_name, agent in self.agents.items():
                        ratio = votes[cur_name]
                        agent.train_predictor(trajectories_a, trajectories_b, ratio)

        for agent in self.agents.values():
            agent.train(number_of_batches, step)

    def update_epsilon(self, step):
       for agent in self.agents.values():
           agent.update_epsilon(step)

    def save_models(self, save_dir, step):
        for name in self.names:
            self.agents[name].save(save_dir, name, step)

    def load_models(self, save_dir, step):
        for name in self.names:
            self.agents[name].load(save_dir, name, step)