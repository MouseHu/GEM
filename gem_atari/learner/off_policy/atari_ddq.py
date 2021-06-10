import copy
import numpy as np
from envs import make_env


class AtariDDQLearner(object):
    def __init__(self, args):
        self.args = args
        self.memory = args.buffer
        self.steps_counter = 0
        self.target_count = 0
        self.learner_info = [
            'Epsilon',
            'Explore_steps'
        ]

        self.rews_sum = 0
        self.rews_cnt = 0
        args.rews_norm = 1.0

        args.eps_act = args.eps_l
        self.eps_decay = (args.eps_l - args.eps_r) / args.eps_decay

        self.sequence = []
        self.beta = self.args.beta
        self.batch_size = self.args.batch_size

    def learn(self, args, env, agent, buffer):
        self.sequence = []
        for _ in range(args.iterations):
            obs = env.get_obs()
            for timestep in range(args.timesteps):
                obs_pre = obs
                action, qs = agent.step_with_q(obs, explore=True, target=True)
                # action = agent.step(obs, explore=True)
                args.eps_act = max(args.eps_r, args.eps_act - self.eps_decay)
                obs, reward, done, _ = env.step(action)
                self.steps_counter += 1
                # frame = env.get_frame()
                # q = 0
                self.sequence.append((obs_pre, action, None, qs, reward, done, False))
                if done:
                    args.logger.add_record('Explore_steps', env.steps)
                    action, qs = agent.step_with_q(obs, explore=True, target=True)
                    self.sequence.append((obs, action, None, qs, 0, True, True))
                    self.memory.update_sequence_with_qs(self.sequence, beta=self.beta)
                    self.sequence = []
                    obs = env.reset()
            # agent.normalizer_update(buffer.sample_batch())
            args.logger.add_record('Epsilon', self.args.eps_act)

            if buffer.steps_counter >= args.warmup:
                for _ in range(args.train_batches):
                    info = agent.train(buffer.sample(self.batch_size, mix=False))
                    args.logger.add_dict(info)
                    self.target_count += 1
                    if self.target_count % args.train_target == 0:
                        agent.target_update()
                        self.memory.update_memory(beta=self.beta)
