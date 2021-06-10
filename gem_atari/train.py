import numpy as np
import time
from common import get_args,experiment_setup
# import horovod.tensorflow as hvd

if __name__=='__main__':
	args = get_args()
	# if args.xian:
	# 	hvd.init()
	env, agent, buffer, learner, tester = experiment_setup(args)

	args.logger.summary_init(agent.graph, agent.sess)

	# Progress info
	args.logger.add_item('Epoch')
	args.logger.add_item('Cycle')
	args.logger.add_item('Episodes@green')
	args.logger.add_item('Timesteps')
	args.logger.add_item('TimeCost(sec)/train')
	args.logger.add_item('TimeCost(sec)/test')

	# Normalizer info
	if not(agent.obs_normalizer is None):
		args.logger.add_item('Norm_mean')
		args.logger.add_item('Norm_std')

	# Algorithm info
	for key in agent.train_info.keys():
		args.logger.add_item(key, 'scalar')
	for key in learner.learner_info:
		args.logger.add_item(key, 'scalar')

	# Test info
	for key in agent.step_info.keys():
		args.logger.add_item(key, 'scalar')
	for key in env.env_info.keys():
		args.logger.add_item(key, 'scalar')
	for key in tester.info:
		args.logger.add_item(key, 'scalar')

	args.logger.summary_setup()

	episodes_cnt = 0
	for epoch in range(args.epoches):
		for cycle in range(args.cycles):
			args.logger.tabular_clear()
			args.logger.summary_clear()

			start_time = time.time()
			learner.learn(args, env, agent, buffer)
			args.logger.add_record('TimeCost(sec)/train', time.time()-start_time)

			start_time = time.time()
			tester.cycle_summary()
			args.logger.add_record('TimeCost(sec)/test', time.time()-start_time)

			args.logger.add_record('Epoch', str(epoch)+'/'+str(args.epoches))
			args.logger.add_record('Cycle', str(cycle)+'/'+str(args.cycles))
			args.logger.add_record('Episodes', buffer.counter)
			args.logger.add_record('Timesteps', learner.steps_counter)
			if not(agent.obs_normalizer is None):
				args.logger.add_record('Norm_mean', np.mean(agent.obs_normalizer.get_mean()))
				args.logger.add_record('Norm_std', np.mean(agent.obs_normalizer.get_std()))

			args.logger.tabular_show(args.tag)
			args.logger.summary_show(buffer.counter)

		tester.epoch_summary()

	tester.final_summary()
