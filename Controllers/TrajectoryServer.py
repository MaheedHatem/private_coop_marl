import numpy as np
from config import Config
from Agents.BaseAgent import BaseAgent

class TrajectoryServer():
    def __init__(self, config, agents, rng = None):
        if rng == None:
            rng = np.random.default_rng()
        self.rng = rng
        self.max_size = config.trajectory_buffer_size
        self.trajectory_length = config.trajectory_length
        self.pairs_count = config.pairs_count
        self.trajectories = np.zeros((self.max_size, self.trajectory_length), dtype=np.int32)
        self.agents = agents
        self.size = 0
        self.curi = 0
        self.curj = 0

    def insert_sample(self, sample_id):
        self.trajectories[self.curi, self.curj] = sample_id
        self.curj += 1
        if self.curj == self.trajectory_length:
            self.curj = 0
            self.curi = (self.curi + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)
    
    def create_trajectories_pairs(self):
        # pairs = np.triu_indices(self.size, k = 1)
        # selected_pairs = self.rng.choice(len(pairs[0]), self.pairs_count, replace=False)
        # return self.trajectories[pairs[0][selected_pairs]], self.trajectories[pairs[1][selected_pairs]]
        indices = self.rng.permutation(self.size)
        return self.trajectories[indices[0:self.pairs_count]], self.trajectories[indices[self.pairs_count:self.pairs_count*2]]

    def get_votes(self):
        trajectories_a, trajectories_b = self.create_trajectories_pairs()
        votes = {}
        for name, agent in self.agents.items():
            votes[name] = agent.compare_trajectories(trajectories_a, trajectories_b)
        return trajectories_a, trajectories_b, votes