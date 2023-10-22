from gym.wrappers import TimeLimit as GymTimeLimit
import gym
from stable_baselines3.common.vec_env import VecEnvWrapper
from gym import ObservationWrapper, spaces
import numpy as np

class TimeLimit(GymTimeLimit):
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env, max_episode_steps)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            if isinstance(done, dict):
                info['TimeLimit.truncated'] = not all(done.values())
                for agent in done.keys():
                    done[agent] = True
            else:
                info['TimeLimit.truncated'] = not all(done)
                done = len(observation) * [True]
        return observation, reward, done, info

class SquashDones(gym.Wrapper):

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

class FlattenObservation(ObservationWrapper):

    def __init__(self, env):
        super(FlattenObservation, self).__init__(env)
        sa_obs = env.observation_space
        flatdim = spaces.flatdim(sa_obs)
        self.observation_space = spaces.Tuple([spaces.Box(
                low=-float("inf"),
                high=float("inf"),
                shape=(flatdim,),
                dtype=np.float32,
            )] * self.env.n_agents)

    def observation(self, observation):
        return tuple([
            spaces.flatten(self.env.observation_space, observation[i])
            for i in range(self.env.n_agents)
        ])


class CPRWrapper(gym.Wrapper):
    def __init__(self, env):
        super(CPRWrapper, self).__init__(env)
    def step(self, action):
        action = {i: a for i, a in enumerate(action) }
        observation, reward, done, info = self.env.step(action)
        return observation, tuple(list(reward.values())), tuple(list(done.values())), info