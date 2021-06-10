import gym
import numpy as np
from utils.os_utils import remove_color

class RAMEnv():
	def __init__(self, args):
		self.args = args
		self.env = gym.make(args.env).env

		self.action_space = self.env.action_space
		self.observation_space = self.env.observation_space
		assert type(self.action_space) is gym.spaces.discrete.Discrete
		self.acts_dims = [self.action_space.n]
		self.obs_dims = list(self.observation_space.shape)

		self.render = self.env.render
		self.get_obs = self.env._get_obs

		self.reset()
		self.env_info = {
			'Steps': self.process_info_steps, # episode steps
			'Rewards@green': self.process_info_rewards # episode cumulative rewards
		}

	def process_info_steps(self, obs, reward, info):
		self.steps += 1
		return self.steps

	def process_info_rewards(self, obs, reward, info):
		self.rewards += reward
		return self.rewards

	def process_info(self, obs, reward, info):
		return {
			remove_color(key): value_func(obs, reward, info)
			for key, value_func in self.env_info.items()
		}

	def step(self, action):
		obs, reward, done, info = self.env.step(action)
		info = self.process_info(obs, reward, info)
		self.last_obs = obs.copy()
		return obs, reward, done, info

	def reset_ep(self):
		self.steps = 0
		self.rewards = 0.0

	def reset(self):
		self.reset_ep()
		self.last_obs = (self.env.reset()).copy()
		return self.last_obs.copy()
