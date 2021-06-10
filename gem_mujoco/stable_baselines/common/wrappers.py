import gym
from gym.core import Wrapper
import numpy as np
from gym.spaces import Box


class NHWCWrapper(Wrapper):
    def __init__(self, env):
        super(NHWCWrapper, self).__init__(env)

        obs_space = env.observation_space
        assert isinstance(obs_space, Box)
        low, high, shape = obs_space.low, obs_space.high, obs_space.shape
        # print("www",low,high,shape)
        new_shape = shape[1:] + (shape[0],)
        low = low.transpose((1,2,0))
        high = high.transpose((1,2,0))
        self.observation_space = Box(low, high, shape=new_shape)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        new_obs = obs.transpose((1, 2, 0))
        return new_obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        new_obs = obs.transpose((1, 2, 0))
        return new_obs


class TimestepWrapper(Wrapper):
    def __init__(self, env, scale=0.01):
        super(TimestepWrapper, self).__init__(env)
        self.scale = scale
        # low = np.append(self.env.observation_space.low, np.array([-np.inf]))
        # high = np.append(self.env.observation_space.high, np.array([np.inf]))
        # self.observation_space = gym.spaces.Box(low, high)
        self.time_step = 0
        self.max_step = env.unwrapped.spec.max_episode_steps
        print("max_step: ",self.max_step)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.time_step += 1
        # obs = np.append(obs, np.array(self.time_step * self.scale))
        if done and self.time_step < self.max_step:
            truly_done = True
        else:
            truly_done = False
        info['truly_done'] = truly_done
        return obs, reward, done, info

    def reset(self):
        self.time_step = 0
        obs = self.env.reset()
        # obs = np.append(obs, np.array(self.time_step * self.scale))
        return obs


class EpisodicRewardWrapper(Wrapper):
    def __init__(self, env):
        super(EpisodicRewardWrapper, self).__init__(env)
        self.cum_reward = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.cum_reward += reward
        reward = self.cum_reward if done else 0
        return obs, reward, done, info

    def reset(self):
        self.cum_reward = 0
        return self.env.reset()


class DelayedRewardWrapper(Wrapper):
    def __init__(self, env, delay=10):
        super(DelayedRewardWrapper, self).__init__(env)
        self.cum_reward = 0
        self.delay = delay
        self.time_step = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.cum_reward += reward
        self.time_step += 1
        if done or self.delay == 0 or (self.time_step % self.delay == 0):
            reward = self.cum_reward
            self.cum_reward = 0
        else:
            reward = 0

        return obs, reward, done, info

    def reset(self):
        self.cum_reward = 0
        self.time_step = 0
        return self.env.reset()
