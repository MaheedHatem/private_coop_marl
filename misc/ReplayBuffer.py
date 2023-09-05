
import numpy as np
import scipy
from collections.abc import Iterable


class ReplayBuffer():
    def __init__(self, obs_dim, num_parallel,
                 max_size, trajectory_buffer_size, batch_size, gamma, rng=None):
        self.max_size = max_size
        self.max_trajectory_buffer_size = trajectory_buffer_size
        self.rng = rng
        self.batch_size = batch_size
        if rng == None:
            rng = np.random.default_rng()
        self.rng = rng
        self.size = 0
        self.cur = 0
        self.num_parallel = num_parallel
        self.cur_traj = 0
        self.obs = np.zeros(
            (max_size, num_parallel, *obs_dim), dtype=np.float32)
        self.act = np.zeros((max_size, num_parallel), dtype=np.int64)
        self.obs_traj = np.zeros(
            (trajectory_buffer_size, num_parallel, *obs_dim), dtype=np.float32)
        self.act_traj = np.zeros(
            (trajectory_buffer_size, num_parallel), dtype=np.int64)
        self.rewards_traj = np.zeros(
            (trajectory_buffer_size, num_parallel), dtype=np.float32)
        self.next_obs = np.zeros(
            (max_size, num_parallel, *obs_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, num_parallel), dtype=np.float32)
        self.val = np.zeros((max_size, num_parallel), dtype=np.float32)
        self.other_val = np.zeros((max_size, num_parallel), dtype=np.float32)
        self.other_rewards = np.zeros(
            (max_size, num_parallel), dtype=np.float32)
        self.gamma = gamma
        self.done = np.zeros((max_size, num_parallel), dtype=np.int64)
        self.truncated = np.zeros((max_size, num_parallel), dtype=np.int64)
        self.ret = np.zeros((max_size, num_parallel))
        self.other_ret = np.zeros((max_size, num_parallel))
        self.sample_id = np.zeros(
            (trajectory_buffer_size, num_parallel), dtype=np.int32)
        self.path_start = 0

    def insert_other_reward(self, reward):
        self.other_rewards[self.cur] = np.squeeze(reward)

    def insert_val(self, val):
        self.val[self.cur] = np.squeeze(val)

    def insert_other_val(self, other_val):
        self.other_val[self.cur] = np.squeeze(other_val)

    def insert(self, obs, act,
               next_obs, reward, done, truncated, sample_id):
        self.obs[self.cur] = obs
        self.act[self.cur] = act
        self.obs_traj[self.cur_traj] = obs
        self.act_traj[self.cur_traj] = act
        self.next_obs[self.cur] = next_obs
        self.rewards[self.cur] = reward
        self.rewards_traj[self.cur_traj] = reward
        self.done[self.cur] = done
        self.truncated[self.cur] = truncated
        self.sample_id[self.cur_traj] = sample_id
        assert sample_id == self.cur_traj
        self.cur = (self.cur + 1) % self.max_size
        self.cur_traj = (self.cur_traj + 1) % self.max_trajectory_buffer_size
        self.size = min(self.size + 1, self.max_size)

    def get_batch(self):
        assert self.size >= self.batch_size
        index = self.rng.choice(self.size, self.batch_size, replace=False)
        batch = self.obs[index], self.act[index], self.next_obs[index], self.rewards[index], self.done[index]
        batch = tuple([self.remove_processes_dimension(dat, 1)
                      for dat in batch])
        return batch

    def get_data(self):
        assert self.size == self.max_size and self.cur == 0
        self.size = 0
        self.cur = 0
        self.compute_returns()
        data = (self.obs, self.act, self.rewards,
                self.ret, self.other_ret)
        data = tuple([self.remove_processes_dimension(dat, 1)
                      for dat in data])
        return data

    def get_rewards(self, trajectories):
        return self.remove_processes_dimension(self.rewards_traj[trajectories].sum(axis=1), 1)

    def get_obs(self, trajectories):
        return self.remove_processes_dimension(self.obs_traj[trajectories], 2)

    def get_act(self, trajectories):
        return self.remove_processes_dimension(self.act_traj[trajectories], 2)

    def compute_returns(self):

        truncated = np.copy(self.truncated)
        truncated[-1] = np.logical_or(np.logical_not(self.done[-1]),
                                      truncated[-1])
        next_val = 0
        next_other_val = 0
        for i in reversed(range(self.max_size)):
            next_val = (1-truncated[i])*(self.rewards[i] + (1-self.done[i]) * self.gamma * next_val) + \
                truncated[i] * (self.rewards[i] + self.gamma * self.val[i])
            next_other_val = (1-truncated[i])*(self.other_rewards[i] + (1-self.done[i]) * self.gamma * next_other_val) + \
                truncated[i] * (self.other_rewards[i] +
                                self.gamma * self.other_val[i])
            self.ret[i] = next_val
            self.other_ret[i] = next_other_val

    def remove_processes_dimension(self, arr, axis):
        assert arr.shape[axis] == self.num_parallel
        arr = np.moveaxis(arr, axis, 0)
        arr = arr.reshape(arr.shape[0]*arr.shape[1], *arr.shape[2:])
        return arr
