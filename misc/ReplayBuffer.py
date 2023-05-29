
import numpy as np
from typing import Tuple


def combined_shape(length: int, shape: np.ndarray = None) -> np.ndarray:
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class ReplayBuffer():
    def __init__(self, obs_dim: Tuple[int],
                 max_size: int, batch_size: int, rng=None):
        self.max_size = max_size
        self.rng = rng
        self.batch_size = batch_size
        if rng == None:
            rng = np.random.default_rng()
        self.rng = rng
        self.size = 0
        self.cur = 0
        self.obs = np.zeros(combined_shape(max_size, obs_dim), dtype=np.float32)
        self.act = np.zeros(max_size, dtype=np.int64)
        self.next_obs = np.zeros(combined_shape(max_size, obs_dim), dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.done = np.zeros(max_size, dtype=np.int64)
        self.sample_id = np.zeros(max_size, dtype=np.int32)

    def insert(self, obs: np.ndarray, act: np.ndarray,
               next_obs: np.ndarray, reward: float, done: int, sample_id: int):
        self.obs[self.cur] = obs
        self.act[self.cur] = act
        self.next_obs[self.cur] = next_obs
        self.rewards[self.cur] = reward
        self.done[self.cur] = done
        self.sample_id[self.cur] = sample_id
        assert sample_id == self.cur
        self.cur = (self.cur + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def get_batch(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert self.size >= self.batch_size
        index = self.rng.choice(self.size, self.batch_size, replace=False)
        return self.obs[index], self.act[index], self.next_obs[index], self.rewards[index], self.done[index]

    def get_rewards(self, trajectories: np.ndarray) -> np.ndarray:
        return self.rewards[trajectories].sum(axis = 1)

    def get_obs(self, trajectories: np.ndarray) -> np.ndarray:
        return self.obs[trajectories]

    def get_act(self, trajectories: np.ndarray) -> np.ndarray:
        return self.act[trajectories]
