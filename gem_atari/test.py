import copy
import numpy as np
from envs import make_env
from utils.os_utils import make_dir

class Tester:
	def __init__(self, args):
		self.args = args
		self.env = make_env(args)
		self.info = []

		if args.save_rews:
			make_dir('log/rews', clear=False)
			self.rews_record = {}
			self.rews_record[args.env] = []

		if args.save_Q:
			make_dir('log/Q_std', clear=False)
			make_dir('log/Q_net', clear=False)
			make_dir('log/Q_ground', clear=False)
			self.Q_std_record, self.Q_net_record, self.Q_ground_record = {}, {}, {}
			self.Q_std_record[args.env], self.Q_net_record[args.env], self.Q_ground_record[args.env] = [], [], []
			self.info += ['Q_error/mean', 'Q_error/std']

	def test_rollouts(self):
		rewards_sum = 0.0
		rews_List, V_pred_List = [], []
		for _ in range(self.args.test_rollouts):
			rewards = 0.0
			rews_list, V_pred_list = [], []
			obs = self.env.reset()
			for timestep in range(self.args.test_timesteps):
				action, info = self.args.agent.step(obs, explore=False, test_info=True)
				assert np.prod(info['Q_average'].shape)==1
				V_pred_list.append(np.sum(info['Q_average']))
				self.args.logger.add_dict(info)
				if self.args.learn[:5]=='atari':
					if np.random.uniform(0.0,1.0)<=self.args.test_eps:
						action = self.env.action_space.sample()
				obs, reward, done, info = self.env.step(action)
				rewards += reward
				rews_list.append(reward)
				if done: break
			rewards_sum += rewards
			rews_List.append(rews_list)
			V_pred_List.append(V_pred_list)
			self.args.logger.add_dict(info)

		if self.args.save_rews:
			step = self.args.learner.steps_counter
			rews = rewards_sum/self.args.test_rollouts
			self.rews_record[self.args.env].append((step, rews))

		if self.args.save_Q:
			err_List = []
			rews_sum, V_pred_sum, record_cnt = 0.0, 0.0, 0
			for rews_list, V_pred_list_zip in zip(rews_List, V_pred_List):
				V_pred_list = V_pred_list_zip
				cnt, V_now = len(rews_list), 0.0
				for i in range(cnt):
					rew_i = rews_list[-i]
					if self.args.learn[:5]=='atari':
						rew_i = np.clip(rew_i, -self.args.rews_scale, self.args.rews_scale)
					V_now = rew_i + self.args.gamma*V_now
					if (i>1.0/(1.0-self.args.gamma)) or (i==cnt-1):
						record_cnt += 1
						rews_sum += V_now
						V_pred_sum += V_pred_list[-i]
						err_List.append(V_pred_list[-i]-V_now)
			if record_cnt!=0:
				rews_sum /= record_cnt
				V_pred_sum /= record_cnt
			step = self.args.learner.steps_counter
			err_List = np.array(err_List)
			Q_error_mean, Q_error_std = np.mean(err_List), np.std(err_List)
			self.Q_std_record[self.args.env].append((step, Q_error_mean, Q_error_std))
			self.Q_net_record[self.args.env].append((step, V_pred_sum))
			self.Q_ground_record[self.args.env].append((step, rews_sum))
			self.args.logger.add_record('Q_error/mean', Q_error_mean)
			self.args.logger.add_record('Q_error/std', Q_error_std)

	def cycle_summary(self):
		self.test_rollouts()

	def epoch_summary(self):
		if self.args.save_rews:
			for key, acc_info in self.rews_record.items():
				log_folder = 'rews'
				if self.args.tag!='': log_folder = log_folder+'/'+self.args.tag
				self.args.logger.save_npz(acc_info, key, log_folder)
		if self.args.save_Q:
			for key, acc_info in self.Q_std_record.items():
				log_folder = 'Q_std'
				if self.args.tag!='': log_folder = log_folder+'/'+self.args.tag
				self.args.logger.save_npz(acc_info, key, log_folder)
			for key, acc_info in self.Q_net_record.items():
				log_folder = 'Q_net'
				if self.args.tag!='': log_folder = log_folder+'/'+self.args.tag
				self.args.logger.save_npz(acc_info, key, log_folder)
			for key, acc_info in self.Q_ground_record.items():
				log_folder = 'Q_ground'
				if self.args.tag!='': log_folder = log_folder+'/'+self.args.tag
				self.args.logger.save_npz(acc_info, key, log_folder)

	def final_summary(self):
		pass
