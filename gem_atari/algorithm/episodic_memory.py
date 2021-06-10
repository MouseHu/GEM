import numpy as np
import os
import pickle as pkl
import copy
from algorithm.segment_tree import SumSegmentTree, MinSegmentTree
import random
import math
import time


def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)


class EpisodicMemory(object):
    def __init__(self, args):
        self.args = args
        buffer_size = int(args.buffer_size)
        self.state_dim = args.state_dim
        self.capacity = buffer_size
        self.curr_capacity = 0
        self.pointer = 0
        self.obs_space = args.obs_dims
        self.action_shape = args.acts_dims
        self.rews_scale = args.rews_scale
        self.frames = args.frames
        self.query_buffer = np.zeros((buffer_size, args.state_dim))
        self._q_values = -np.inf * np.ones(buffer_size + 1)
        self.returns = -np.inf * np.ones(buffer_size + 1)
        print(self.obs_space)
        self.frame_buffer = np.empty([buffer_size, ] + self.obs_space[:-1], np.uint8)
        self.begin_obs = dict()
        # self.replay_buffer = np.empty([buffer_size+buffer_size/10, ] + self.obs_space[:-1], np.uint8)
        self.action_buffer = np.empty([buffer_size, 1], np.int)
        self.reward_buffer = np.empty((buffer_size,), np.float32)
        self.steps = np.empty((buffer_size,), np.int)
        self.episode_steps = np.zeros((buffer_size,), np.int)
        self.done_buffer = np.empty((buffer_size,), np.bool)
        self.truly_done_buffer = np.empty((buffer_size,), np.bool)
        self.next_id = -1 * np.ones(buffer_size)
        self.prev_id = [[] for _ in range(buffer_size)]
        self.ddpg_q_values = -np.inf * np.ones(buffer_size)
        self.contra_count = np.ones((buffer_size,))
        self.lru = np.zeros(buffer_size)
        self.time = 0
        self.gamma = args.gamma
        # self.hashes = dict()
        self.reward_mean = None
        self.min_return = 0
        self.end_points = []
        assert args.alpha > 0
        self._alpha = args.alpha
        self.beta_set = [-1]
        self.beta_coef = [1.]
        it_capacity = 1
        while it_capacity < buffer_size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

        self.q_func = None
        self.repr_func = None
        self.obs_ph = None
        self.action_ph = None
        self.sess = None
        self.steps_counter = 0
        self.counter = 0

        self.sequence = []
        self.batch_size = args.batch_size * 16

    def get_ring_obs(self, idx, num_frame):
        begin_idx = (idx + num_frame) % self.capacity
        if begin_idx < idx:
            obs = np.concatenate([self.frame_buffer[idx:], self.frame_buffer[:begin_idx]], axis=0)
        else:
            obs = self.frame_buffer[idx:begin_idx, :, :]

        obs = np.moveaxis(obs, 0, -1)
        obs = obs[:, :, ::-1]
        return obs

    def get_obs(self, idxes):
        return np.array([self._get_obs(idx) for idx in idxes])

    def _get_obs(self, idx):
        if self.episode_steps[idx] + 1 < self.frames:
            # need to append first
            first_idx = (idx + self.episode_steps[idx]) % self.capacity
            try:
                begin_obs = self.begin_obs[first_idx][:, :, self.episode_steps[idx]:-1]
            except KeyError:
                print(idx, self.episode_steps[idx], first_idx)
                print(list(self.begin_obs.keys()))
                exit(-1)
            end_obs = self.get_ring_obs(idx, self.episode_steps[idx] + 1)
            obs = np.concatenate([begin_obs, end_obs], axis=-1)
        else:
            obs = self.get_ring_obs(idx, self.frames)
        return obs

    def update_func(self, agent):
        self.q_func = agent.em_q
        self.repr_func = None
        self.obs_ph = agent.em_raw_obs_ph
        self.sess = agent.sess

    def clean(self):
        buffer_size, state_dim, obs_space, action_shape = self.capacity, self.state_dim, self.obs_space, self.action_shape
        self.curr_capacity = 0
        self.pointer = 0

        self.query_buffer = np.zeros((buffer_size, state_dim))
        self._q_values = -np.inf * np.ones(buffer_size + 1)
        # self.return_buffer = -np.inf * np.ones(buffer_size + 1)
        self.returns = -np.inf * np.ones(buffer_size + 1)
        # self.replay_buffer = np.empty((buffer_size,) + obs_space.shape, np.float32)
        self.frame_buffer = np.empty((buffer_size,) + obs_space.shape[:-1], np.float32)
        self.begin_obs = dict()
        self.action_buffer = np.empty((buffer_size,) + action_shape, np.uint8)
        self.reward_buffer = np.empty((buffer_size,), np.float32)
        self.steps = np.empty((buffer_size,), np.int)
        self.done_buffer = np.empty((buffer_size,), np.bool)
        self.truly_done_buffer = np.empty((buffer_size,), np.bool)
        self.next_id = -1 * np.ones(buffer_size)
        self.prev_id = [[] for _ in range(buffer_size)]
        self.ddpg_q_values = -np.inf * np.ones(buffer_size)
        self.contra_count = np.ones((buffer_size,))
        self.lru = np.zeros(buffer_size)
        self.time = 0

        it_capacity = 1
        while it_capacity < buffer_size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0
        self.end_points = []

    @property
    def q_values(self):
        return self._q_values

    def squeeze(self, obses):
        return np.array([(obs - self.obs_space.low) / (self.obs_space.high - self.obs_space.low) for obs in obses])

    def unsqueeze(self, obses):
        return np.array([obs * (self.obs_space.high - self.obs_space.low) + self.obs_space.low for obs in obses])

    def save(self, filedir):
        save_dict = {"query_buffer": self.query_buffer, "returns": self.returns,
                     "frame_buffer": self.frame_buffer, "begin_obs": self.begin_obs,
                     "reward_buffer": self.reward_buffer,
                     "truly_done_buffer": self.truly_done_buffer, "next_id": self.next_id, "prev_id": self.prev_id,
                     "gamma": self.gamma, "_q_values": self._q_values, "done_buffer": self.done_buffer,
                     "curr_capacity": self.curr_capacity, "capacity": self.capacity}
        with open(os.path.join(filedir, "episodic_memory.pkl"), "wb") as memory_file:
            pkl.dump(save_dict, memory_file)

    def add(self, obs, action, state, sampled_return, next_id=-1):

        index = self.pointer
        self.pointer = (self.pointer + 1) % self.capacity

        if self.curr_capacity >= self.capacity:
            # Clean up old entry
            if index in self.end_points:
                self.end_points.remove(index)
                for prev_id in self.prev_id[index]:
                    if prev_id not in self.end_points:
                        self.end_points.append(prev_id)
            self.prev_id[index] = []
            self.next_id[index] = -1
            self.q_values[index] = -np.inf
            if self.episode_steps[index] == 0:
                del self.begin_obs[index]
            self.episode_steps[index] = 0
        else:
            self.curr_capacity = min(self.capacity, self.curr_capacity + 1)
        # Store new entry
        # self.replay_buffer[index] = obs
        self.frame_buffer[index] = obs[:, :, -1]
        self.action_buffer[index] = action
        if state is not None:
            self.query_buffer[index] = state
        self.q_values[index] = sampled_return
        # self.returns[index] = sampled_return
        self.lru[index] = self.time

        if next_id >= 0:
            self.next_id[index] = next_id
            if index not in self.prev_id[next_id]:
                self.prev_id[next_id].append(index)
        self.time += 0.01
        self.steps_counter += 1

        return index

    def update_priority(self):
        priorities = 1 / np.sqrt(self.contra_count[:self.curr_capacity])
        priorities = priorities / np.max(priorities)
        for idx, priority in enumerate(priorities):
            priority = max(priority, 1e-6)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)

    def sample_neg_keys(self, avoids, batch_size):
        # sample negative keys
        assert batch_size + len(
            avoids) <= self.capacity, "can't sample that much neg samples from episodic memory!"
        places = []
        while len(places) < batch_size:
            ind = np.random.randint(0, self.curr_capacity)
            if ind not in places:
                places.append(ind)
        return places

    def compute_approximate_return(self, obses, actions=None):
        returns = []
        num_epoch = int(np.ceil(len(obses)/self.batch_size))
        for e in range(num_epoch):
            low = e*self.batch_size
            high = min(len(obses),(e+1)*self.batch_size)
            batch_obs = obses[low:high].astype(np.float32) / 255.0
            returns.append(np.array(self.sess.run(self.q_func, feed_dict={self.obs_ph: batch_obs})).reshape(-1,1))
        return np.concatenate(returns)

        # return np.array(self.sess.run(self.q_func, feed_dict={self.obs_ph: obses.astype(np.float32) / 255.0}))

    def compute_statistics(self, batch_size=1024):
        estimated_qs = []
        for i in range(math.ceil(self.curr_capacity / batch_size)):
            start = i * batch_size
            end = min((i + 1) * batch_size, self.curr_capacity)
            obses = self.get_obs([idx for idx in range(start, end)])
            actions = None
            estimated_qs.append(self.compute_approximate_return(obses, actions).reshape(-1))
        estimated_qs = np.concatenate(estimated_qs)
        diff = estimated_qs - self.q_values[:self.curr_capacity]
        return np.min(diff), np.mean(diff), np.max(diff)

    def retrieve_trajectories(self):
        trajs = []
        for e in self.end_points:
            traj = []
            prev = e
            while prev is not None:
                traj.append(prev)
                try:
                    prev = self.prev_id[prev][0]
                    # print(e,prev)
                except IndexError:
                    prev = None
            # print(np.array(traj))
            trajs.append(np.array(traj))
        return trajs

    def update_memory(self, q_base=0, use_knn=False, beta=-1):

        trajs = self.retrieve_trajectories()
        for traj in trajs:
            # print(np.array(traj))
            approximate_qs = self.compute_approximate_return(self.get_obs(traj), self.action_buffer[traj])
            approximate_qs = approximate_qs.reshape(-1)
            assert len(approximate_qs) == len(traj), "length mismatch:{} v.s. {}".format(len(approximate_qs), len(traj))
            approximate_qs = np.insert(approximate_qs, 0, 0)

            self.q_values[traj] = 0
            Rtn = -1e10 if beta < 0 else 0
            for i, s in enumerate(traj):
                approximate_q = self.reward_buffer[s] + \
                                self.gamma * (1 - self.truly_done_buffer[s]) * (approximate_qs[i] - q_base)
                Rtn = self.reward_buffer[s] + self.gamma * (1 - self.truly_done_buffer[s]) * Rtn
                if beta < 0:
                    Rtn = max(Rtn, approximate_q)
                else:
                    Rtn = beta * Rtn + (1 - beta) * approximate_q
                self.q_values[s] = Rtn

    def update_sequence_with_qs(self, sequence, beta=-1):
        # print(sequence)
        next_id = -1
        Rtn = 0
        tRtn = 0
        episode_steps = len(sequence)
        qs = [transition[3] for transition in reversed(sequence)]
        qs = [np.zeros(1)] + qs[:-1]
        ids = []
        for i, transition in enumerate(reversed(sequence)):
            obs, a, z, q_t, r, truly_done, done = transition
            # print(np.mean(z))
            episode_steps = episode_steps - 1
            r = np.clip(r, -self.rews_scale, self.rews_scale)
            if truly_done:
                Rtn = r
                tRtn = r
            else:
                # Rtn = self.gamma * (1-truly_done)*max(Rtn, qs[i]) + r
                if beta < 0:
                    Rtn = self.gamma * max(Rtn, qs[i]) + r
                else:
                    assert beta <= 1
                    Rtn = self.gamma * (beta * Rtn + (1 - beta) * qs[i]) + r
                tRtn = self.gamma * tRtn + r
            current_id = self.add(obs, a, z, Rtn, next_id)
            ids.append(current_id)
            if done:
                self.end_points.append(current_id)
            if episode_steps == 0:
                self.begin_obs[current_id] = obs
            self.frame_buffer[current_id] = obs[:, :, -1]
            self.reward_buffer[current_id] = r
            self.returns[current_id] = tRtn
            self.done_buffer[current_id] = done
            self.truly_done_buffer[current_id] = truly_done
            self.episode_steps[current_id] = episode_steps
            next_id = int(current_id)

        for i, transition in enumerate(reversed(sequence)):
            obs, a, z, q_t, r, truly_done, done = transition
            diff = obs - self.get_obs([ids[i]])[0]
            assert np.sum(abs(diff)) == 0, np.sum(abs(diff))
        # self.update_priority()
        self.counter += 1
        return

    def sample_negative(self, batch_size, batch_idxs, batch_idxs_next, batch_idx_pre):
        neg_batch_idxs = []
        i = 0
        while i < batch_size:
            neg_idx = np.random.randint(0, self.curr_capacity - 2)
            if neg_idx != batch_idxs[i] and neg_idx != batch_idxs_next[i] and neg_idx not in batch_idx_pre[i]:
                neg_batch_idxs.append(neg_idx)
                i += 1
        neg_batch_idxs = np.array(neg_batch_idxs)
        return neg_batch_idxs, self.get_obs(neg_batch_idxs)

    def sample_batch(self):
        return self.sample(self.args.batch_size, mix=False)

    def store_transition(self, transition):

        obs_pre, obs, action, reward, done = transition['obs'], transition['obs_next'], transition['acts'], transition[
            'rews'], transition['done']
        self.sequence.append((obs_pre, action, None, 0, reward, done, False))
        if done:
            self.sequence.append((obs, 0, None, 0, reward, done, True))
            self.update_sequence_with_qs(self.sequence, beta=1)
            self.sequence = []

    @staticmethod
    def switch_first_half(obs0, obs1, batch_size):
        tmp = copy.copy(obs0[:batch_size // 2, ...])
        obs0[:batch_size // 2, ...] = obs1[:batch_size // 2, ...]
        obs1[:batch_size // 2, ...] = tmp
        return obs0, obs1

    def sample(self, batch_size, mix=False, priority=False):
        # Draw such that we always have a proceeding element
        if self.curr_capacity < batch_size + len(self.end_points):
            return None
        if priority:
            self.update_priority()
        batch_idxs = []
        batch_idxs_next = []
        count = 0
        while len(batch_idxs) < batch_size:
            if priority:
                mass = random.random() * self._it_sum.sum(0, self.curr_capacity)
                rnd_idx = self._it_sum.find_prefixsum_idx(mass)
            else:
                rnd_idx = np.random.randint(0, self.curr_capacity)
            count += 1
            assert count < 1e8
            if self.next_id[rnd_idx] == -1:
                continue
                # be careful !!!!!! I use random id because in our implementation obs1 is never used
                # if len(self.prev_id[rnd_idx]) > 0:
                #     batch_idxs_next.append(self.prev_id[rnd_idx][0])
                # else:
                #     batch_idxs_next.append(0)
            else:
                batch_idxs_next.append(self.next_id[rnd_idx])
                batch_idxs.append(rnd_idx)

        batch_idxs = np.array(batch_idxs).astype(np.int)
        batch_idxs_next = np.array(batch_idxs_next).astype(np.int)
        # batch_idx_pre = [self.prev_id[id] for id in batch_idxs]

        obs0_batch = self.get_obs(batch_idxs)
        obs1_batch = self.get_obs(batch_idxs_next)
        # batch_idxs_neg, obs2_batch = self.sample_negative(batch_size, batch_idxs, batch_idxs_next, batch_idx_pre)
        action_batch = self.action_buffer[batch_idxs]
        action1_batch = self.action_buffer[batch_idxs_next]
        # action2_batch = self.action_buffer[batch_idxs_neg]
        reward_batch = self.reward_buffer[batch_idxs]
        terminal1_batch = self.truly_done_buffer[batch_idxs]
        q_batch = self.q_values[batch_idxs]
        return_batch = self.returns[batch_idxs]

        if mix:
            obs0_batch, obs1_batch = self.switch_first_half(obs0_batch, obs1_batch, batch_size)
        if priority:
            self.contra_count[batch_idxs] += 1
            self.contra_count[batch_idxs_next] += 1

        result = {
            'index0': array_min2d(batch_idxs),
            'index1': array_min2d(batch_idxs_next),
            # 'index2': array_min2d(batch_idxs_neg),
            'obs0': array_min2d(obs0_batch).astype(np.float32) / 255.0,
            'obs1': array_min2d(obs1_batch).astype(np.float32) / 255.0,
            # 'obs2': array_min2d(obs2_batch),
            'rewards': array_min2d(reward_batch),
            'actions': array_min2d(action_batch),
            'actions1': array_min2d(action1_batch),
            # 'actions2': array_min2d(action2_batch),
            'count': array_min2d(self.contra_count[batch_idxs] + self.contra_count[batch_idxs_next]),
            'terminals1': array_min2d(terminal1_batch),
            'return': array_min2d(q_batch),
            'true_return': array_min2d(return_batch),
        }
        return result
