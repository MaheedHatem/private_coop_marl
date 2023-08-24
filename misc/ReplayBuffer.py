
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



class ReplayBuffer():
    def __init__(self, obs_dim: Tuple[int], num_parallel: int, 
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
        self.num_parallel = num_parallel
        self.cur_traj = 0
        self.obs = np.zeros((max_size, num_parallel, *obs_dim), dtype=np.float32)
        self.act = np.zeros((max_size, num_parallel), dtype=np.int64)
        self.obs_traj = np.zeros((trajectory_buffer_size, num_parallel, *obs_dim), dtype=np.float32)
        self.act_traj = np.zeros((trajectory_buffer_size, num_parallel), dtype=np.int64)
        self.rewards_traj = np.zeros((trajectory_buffer_size, num_parallel), dtype=np.float32)
        self.next_obs = np.zeros((max_size, num_parallel, *obs_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, num_parallel), dtype=np.float32)
        self.val = np.zeros((max_size, num_parallel), dtype=np.float32)
        self.other_val = np.zeros((max_size, num_parallel), dtype=np.float32)
        self.other_rewards = np.zeros((max_size, num_parallel), dtype=np.float32)
        self.gamma = gamma
        self.done = np.zeros((max_size, num_parallel), dtype=np.int64)
        self.truncated = np.zeros((max_size, num_parallel), dtype=np.int64)
        self.ret = np.zeros((max_size, num_parallel))
        self.other_ret = np.zeros((max_size, num_parallel)) 
        self.sample_id = np.zeros((trajectory_buffer_size, num_parallel), dtype=np.int32)
        self.path_start = 0

    def insert_other_reward(self, reward):
        self.other_rewards[self.cur] = np.squeeze(reward)

    def insert_val(self, val):
        self.val[self.cur] = np.squeeze(val)

    def insert_other_val(self, other_val):
        self.other_val[self.cur] = np.squeeze(other_val)

    def insert(self, obs: np.ndarray, act: np.ndarray,
               next_obs: np.ndarray, reward: float, done: int, truncated: bool, sample_id: int):
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

    def get_batch(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert self.size >= self.batch_size
        index = self.rng.choice(self.size, self.batch_size, replace=False)
        batch = self.obs[index], self.act[index], self.next_obs[index], self.rewards[index], self.done[index]
        batch = tuple([self.remove_processes_dimension(dat, 1) for dat in batch])
        return batch

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert self.size == self.max_size and self.cur == 0
        self.size = 0
        self.cur = 0
        batches_count = int(self.max_size/self.batch_size)
        batches = []
        self.compute_returns()
        for i in range(batches_count):
            start_index = i*self.batch_size
            end_index = (i+1)*self.batch_size
            batch = (self.obs[start_index:end_index], self.act[start_index:end_index], self.rewards[start_index:end_index],
                self.ret[start_index:end_index], self.other_ret[start_index:end_index])
            batch = tuple([self.remove_processes_dimension(dat, 1) for dat in batch])
            batches.append(batch)
        return batches

    def get_rewards(self, trajectories: np.ndarray) -> np.ndarray:
        return self.remove_processes_dimension(self.rewards_traj[trajectories].sum(axis = 1), 1)

    def get_obs(self, trajectories: np.ndarray) -> np.ndarray:
        return self.remove_processes_dimension(self.obs_traj[trajectories], 2)

    def get_act(self, trajectories: np.ndarray) -> np.ndarray:
        return self.remove_processes_dimension(self.act_traj[trajectories], 2)

    def finish_path(self, last_val = 0):
        both_rewards = False
        if(isinstance(last_val, Iterable)):
            assert len(last_val) == 2
            other_val = last_val[1]
            last_val = last_val[0]
            both_rewards = True
        path_end = self.cur
        if self.cur == 0 and self.size == self.max_size:
            path_end = self.max_size
        path_slice = slice(self.path_start, path_end)
        rews = np.append(self.rewards[path_slice], last_val)
        if both_rewards:
            other_rews = np.append(self.other_rewards[path_slice], other_val)
            self.other_ret[path_slice] = discount_cumsum(other_rews, self.gamma)[:-1]
        
        
        self.ret[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start = self.cur

    def compute_returns(self):

        truncated = np.copy(self.truncated)
        truncated[-1] = np.invert(self.done[-1])
        next_val = 0
        next_other_val = 0
        for i in reversed(range(self.max_size)):
            next_val = (1-truncated[i])*(self.rewards[i] + (1-self.done[i]) * self.gamma * next_val) + \
                            truncated[i] * (self.rewards[i] + self.gamma * self.val[i])
            next_other_val = (1-truncated[i])*(self.other_rewards[i] + (1-self.done[i]) * self.gamma * next_other_val) + \
                            truncated[i] * (self.other_rewards[i] + self.gamma * self.other_val[i])
            self.ret[i] = next_val
            self.other_ret[i] = next_other_val


    def remove_processes_dimension(self, arr: np.ndarray, axis):
        assert arr.shape[axis] == self.num_parallel
        arr = np.moveaxis(arr, axis, 0)
        arr = arr.reshape(arr.shape[0]*arr.shape[1],*arr.shape[2:])
        return arr