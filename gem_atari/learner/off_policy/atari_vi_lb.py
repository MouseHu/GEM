import copy
import numpy as np
from envs import make_env
from algorithm.hash import Hash

class DiscountedQueue:
    def __init__(self, args):
        self.args = args
        self.buffer = args.buffer
        self.transitions = []
        self.H = int(1.0/(1-args.gamma))

    def store_transition(self, transition):
        self.transitions.append(transition)
        if transition['done']:
            while len(self.transitions)>0:
                self.pop()
        elif len(self.transitions)>self.H:
            self.pop()

    def pop(self):
        transition_pop = copy.deepcopy(self.transitions[0])
        rets = 0.0
        for transition in self.transitions[::-1]:
            rews = np.clip(transition['rews'], -self.args.rews_scale, self.args.rews_scale)
            rets = rews+self.args.gamma*rets
        transition_pop['hash_next'] = self.args.hash_lib.get_hash(transition_pop['obs_next'])
        self.args.hash_lib.add_return(self.args.hash_lib.get_hash(transition_pop['obs']), rets)
        self.buffer.store_transition(transition_pop)
        self.transitions.pop(0)

class AtariValueIterationLowerBoundedLearner:
    def __init__(self, args):
        self.args = args
        self.queue = DiscountedQueue(args)
        self.steps_counter = 0
        self.target_count = 0
        self.learner_info = [
            'Epsilon',
            'Explore_steps',
            'Hash_states'
        ]

        self.rews_sum = 0
        self.rews_cnt = 0
        args.rews_norm = 1.0

        args.eps_act = args.eps_l
        self.eps_decay = (args.eps_l-args.eps_r)/args.eps_decay

        args.hash_lib = Hash(args)

    def learn(self, args, env, agent, buffer):
        for _ in range(args.iterations):
            obs = env.get_obs()
            for timestep in range(args.timesteps):
                obs_pre = obs
                action = agent.step(obs, explore=True)
                args.eps_act = max(args.eps_r, args.eps_act-self.eps_decay)
                obs, reward, done, _ = env.step(action)
                self.steps_counter += 1
                frame = env.get_frame()
                transition = {
                    'obs': obs_pre,
                    'obs_next': obs,
                    'frame_next': frame,
                    'acts': action,
                    'rews': reward,
                    'done': done
                }
                self.queue.store_transition(transition)
                if done:
                    args.logger.add_record('Explore_steps', env.steps)
                    obs = env.reset()
            # agent.normalizer_update(buffer.sample_batch())
            args.logger.add_record('Epsilon', self.args.eps_act)
            args.logger.add_record('Hash_states', self.args.hash_lib.get_state_tot())

            if buffer.steps_counter>=args.warmup:
                for _ in range(args.train_batches):
                    batch = buffer.sample_batch()
                    batch['rets'] = []
                    for r, hash_next in zip(batch['rews'], batch['hash_next']):
                        avg_ret = r[0] + (self.args.gamma**self.args.nstep) * self.args.hash_lib.get_return(hash_next)
                        batch['rets'].append([avg_ret])
                    info = agent.train(batch)
                    if self.target_count%args.train_target!=0:
                        info.pop('Q_L1_loss')
                    args.logger.add_dict(info)
                    self.target_count += 1
                    if self.target_count%args.train_target==0:
                        agent.target_update()
