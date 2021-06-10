import numpy as np
from envs import make_env
import math

def GAE(args, agent, seg):
	pool = dict(obs=[], acts=[], rets=[], advs=[])
	last_gae, last_v_pred = 0.0, agent.get_v_pred(seg[-1]['obs_next'])
	for info in seg[::-1]:
		done = np.float32(info['done'])
		delta = info['rews']+(1.0-done)*args.gamma*last_v_pred-info['v_pred']
		last_gae = delta+(1.0-done)*args.gamma*args.lam*last_gae
		last_v_pred = info['v_pred']
		pool['obs'].append(info['obs'].copy())
		pool['acts'].append(info['acts'].copy())
		pool['rets'].append(last_gae+last_v_pred)
		pool['advs'].append(last_gae)
	for key in pool.keys():
		pool[key] = np.array(pool[key])
	pool['advs'] = (pool['advs']-np.mean(pool['advs'],axis=0))/(np.std(pool['advs'],axis=0)+1e-6)

	return pool

class NormalLearner:
	def __init__(self, args):
		self.target_count = 0
		self.learner_info = [
			'Explore/steps',
			'Explore/rewards@green'
		]

		assert args.warmup==0

	def learn(self, args, env, agent, buffer):
		for _ in range(args.iterations):
			seg = []
			obs = env.get_obs()
			for timestep in range(args.timesteps):
				obs_pre = obs
				action, v_pred = agent.step(obs, explore=True)
				obs, reward, done, info = env.step(action)
				transition = {
					'obs': obs_pre,
					'obs_next': obs,
					'acts': action,
					'rews': reward,
					'done': done,
					'v_pred': v_pred,
					'info': info
				}
				seg.append(transition)
				buffer.add_counter(transition)
				if done:
					args.logger.add_record('Explore/steps', env.steps)
					args.logger.add_record('Explore/rewards', env.rewards)
					obs = env.reset()

			pool = GAE(args, agent, seg)

			cnt = args.timesteps
			agent.pi_old_update()
			for epoch_id in range(args.batch_epoch):
				perm = np.arange(cnt)
				np.random.shuffle(perm)
				for batch_id in range(cnt//args.batch_size):
					idxs = perm[args.batch_size*batch_id:args.batch_size*(batch_id+1)]
					batch = { key:pool[key][idxs] for key in pool.keys() }
					if epoch_id==0: agent.normalizer_update(batch)
					info = agent.train(batch)
					args.logger.add_dict(info)