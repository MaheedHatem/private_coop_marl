
import numpy as np
from typing import Tuple
import scipy
from collections.abc import Iterable

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def combined_shape(length: int, shape: np.ndarray = None) -> np.ndarray:
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class ReplayBuffer():
    def __init__(self, obs_dim: Tuple[int],
                 max_size: int, trajectory_buffer_size: int, batch_size: int, gamma: float, rng=None):
        self.max_size = max_size
        self.max_trajectory_buffer_size = trajectory_buffer_size
        self.rng = rng
        self.batch_size = batch_size
        if rng == None:
            rng = np.random.default_rng()
        self.rng = rng
        self.size = 0
        self.cur = 0
        self.cur_traj = 0
        self.obs = np.zeros(combined_shape(max_size, obs_dim), dtype=np.float32)
        self.act = np.zeros(max_size, dtype=np.int64)
        self.obs_traj = np.zeros(combined_shape(trajectory_buffer_size, obs_dim), dtype=np.float32)
        self.act_traj = np.zeros(trajectory_buffer_size, dtype=np.int64)
        self.rewards_traj = np.zeros(trajectory_buffer_size, dtype=np.float32)
        self.next_obs = np.zeros(combined_shape(max_size, obs_dim), dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.other_rewards = np.zeros(max_size, dtype=np.float32)
        self.gamma = gamma
        self.done = np.zeros(max_size, dtype=np.int64)
        self.ret = np.zeros(max_size)
        self.other_ret = np.zeros(max_size) 
        self.sample_id = np.zeros(trajectory_buffer_size, dtype=np.int32)
        self.path_start = 0

    def insert_other_reward(self, reward):
        self.other_rewards[self.cur] = reward

    def insert(self, obs: np.ndarray, act: np.ndarray,
               next_obs: np.ndarray, reward: float, done: int, sample_id: int):
        self.obs[self.cur] = obs
        self.act[self.cur] = act
        self.obs_traj[self.cur_traj] = obs
        self.act_traj[self.cur_traj] = act
        self.next_obs[self.cur] = next_obs
        self.rewards[self.cur] = reward
        self.rewards_traj[self.cur_traj] = reward
        self.done[self.cur] = done
        self.sample_id[self.cur_traj] = sample_id
        assert sample_id == self.cur_traj
        self.cur = (self.cur + 1) % self.max_size
        self.cur_traj = (self.cur_traj + 1) % self.max_trajectory_buffer_size
        self.size = min(self.size + 1, self.max_size)

    def get_batch(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert self.size >= self.batch_size
        index = self.rng.choice(self.size, self.batch_size, replace=False)
        return self.obs[index], self.act[index], self.next_obs[index], self.rewards[index], self.done[index]

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert self.size == self.max_size
        self.size = 0
        self.cur = 0
        batches_count = int(self.max_size/self.batch_size)
        batches = []
        for i in range(batches_count):
            start_index = i*self.batch_size
            end_index = (i+1)*self.batch_size
            batches.append((self.obs[start_index:end_index], self.act[start_index:end_index], self.rewards[start_index:end_index],
                self.done[start_index:end_index], self.ret[start_index:end_index], self.other_ret[start_index:end_index]))
        return batches

    def get_rewards(self, trajectories: np.ndarray) -> np.ndarray:
        return self.rewards_traj[trajectories].sum(axis = 1)

    def get_obs(self, trajectories: np.ndarray) -> np.ndarray:
        return self.obs_traj[trajectories]

    def get_act(self, trajectories: np.ndarray) -> np.ndarray:
        return self.act_traj[trajectories]

    def finish_path(self, last_val = 0):
        both_rewards = False
        if(isinstance(last_val, Iterable)):
            assert len(last_val) == 2
            other_val = last_val[1]
            last_val = last_val[0]
            both_rewards = True
        path_slice = slice(self.path_start, self.cur)
        rews = np.append(self.rewards[path_slice], last_val)
        if both_rewards:
            other_rews = np.append(self.other_rewards[path_slice], other_val)
            self.other_ret[path_slice] = discount_cumsum(other_rews, self.gamma)[:-1]
        
        
        self.ret[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start = self.cur
