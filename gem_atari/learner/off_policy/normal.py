import numpy as np
from envs import make_env

class NormalLearner:
	def __init__(self, args):
		self.target_count = 0
		self.learner_info = []

	def learn(self, args, env, agent, buffer):
		for _ in range(args.iterations):
			obs = env.get_obs()
			for timestep in range(args.timesteps):
				obs_pre = obs
				action = agent.step(obs, explore=True)
				obs, reward, done, info = env.step(action)
				transition = {
					'obs': obs_pre,
					'obs_next': obs,
					'acts': action,
					'rews': reward,
					'done': done
				}
				buffer.store_transition(transition)
				if done: obs = env.reset()
			agent.normalizer_update(buffer.sample_batch())

			if buffer.steps_counter>=args.warmup:
				for _ in range(args.train_batches):
					info = agent.train(buffer.sample_batch())
					args.logger.add_dict(info)
					self.target_count += 1
					if self.target_count%args.train_target==0:
						agent.target_update()