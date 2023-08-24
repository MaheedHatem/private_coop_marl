from gym.wrappers import TimeLimit as GymTimeLimit
import gym
from stable_baselines3.common.vec_env import VecEnvWrapper
import numpy as np

class TimeLimit(GymTimeLimit):
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None
        self.wala3 = 0

    def step(self, action):
        assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info['TimeLimit.truncated'] = not all(done)
            done = len(observation) * [True]
        return observation, reward, done, info

class SquashDones(gym.Wrapper):
    r"""Wrapper that squashes multiple dones to a single one using all(dones)"""

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, reward, all(done), info

class SplitAgentsDataWrapper(VecEnvWrapper):

    def __init__(self, venv):
        super().__init__(venv)

    def reset(self):
        return np.array(self.venv.reset())

    def step_async(self, actions):
        return self.venv.step_async(actions)

    def step_wait(self):
        obs, rew, done, info = self.venv.step_wait()
        return np.array(obs), rew.T, done.T, info