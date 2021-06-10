import numpy as np

from stable_baselines.td3.episodic_memory import EpisodicMemory
import threading


class EpisodicMemoryTBP(EpisodicMemory):
    def __init__(self, buffer_size, state_dim, action_shape, obs_space, q_func, repr_func, obs_ph, action_ph, sess,
                 gamma=0.99,
                 alpha=0.6, max_step=1000):
        super(EpisodicMemoryTBP, self).__init__(buffer_size, state_dim, action_shape, obs_space, q_func, repr_func,
                                                obs_ph, action_ph, sess,
                                                gamma, alpha, max_step)
        del self._q_values
        self._q_values = -np.inf * np.ones((buffer_size + 1, 2))
        self.approximate_q_values = -np.inf * np.ones((buffer_size, 4))
        self.batch_size = 16384
        self.max_step = max_step

    def compute_approximate_return_double(self, obses, actions=None):
        return np.array(self.sess.run(self.q_func, feed_dict={self.obs_ph: obses}))

    def compute_approximate_return_double_batch(self, batch_size):
        # batch updating to speed up memory update
        num_updates = int(np.ceil(self.curr_capacity / batch_size))
        # self.approximate_q_values = -np.inf * np.ones((self.curr_capacity, 4))
        for i in range(num_updates):
            start = i * batch_size
            end = min(self.curr_capacity, (i + 1) * batch_size)
            obses = self.replay_buffer[start:end]
            q_values = np.array(self.sess.run(self.q_func, feed_dict={self.obs_ph: obses}))
            q_values = q_values.squeeze().T
            self.approximate_q_values[start:end, :] = q_values

    def update_traj(self, traj, q_base, beta):
        discount_beta = beta ** np.arange(self.max_step)
        approximate_qs = self.approximate_q_values[traj].T
        num_q = len(approximate_qs)
        if num_q >= 4:
            approximate_qs = approximate_qs.reshape((2, num_q // 2, -1))
            approximate_qs = np.min(approximate_qs, axis=1)  # clip double q

        else:
            assert num_q == 2
            approximate_qs = approximate_qs.reshape(2, -1)
        approximate_qs = np.concatenate([np.zeros((2, 1)), approximate_qs], axis=1)
        self.q_values[traj] = 0

        rtn_1 = np.zeros((len(traj), len(traj)))
        rtn_2 = np.zeros((len(traj), len(traj)))

        for i, s in enumerate(traj):
            rtn_1[i, 0], rtn_2[i, 0] = self.reward_buffer[s] + \
                                       self.gamma * (1 - self.truly_done_buffer[s]) * (
                                               approximate_qs[:, i] - q_base)
        for i, s in enumerate(traj):
            rtn_1[i, 1:] = self.reward_buffer[s] + self.gamma * rtn_1[i - 1, :-1]
            rtn_2[i, 1:] = self.reward_buffer[s] + self.gamma * rtn_2[i - 1, :-1]

        if beta > 0:

            double_rtn = [
                [np.dot(rtn_2[i, :min(i + 1, self.max_step)], discount_beta[:min(i + 1, self.max_step)]) / np.sum(
                    discount_beta[:min(i + 1, self.max_step)]),
                 np.dot(rtn_1[i, :min(i + 1, self.max_step)], discount_beta[:min(i + 1, self.max_step)]) / np.sum(
                     discount_beta[:min(i + 1, self.max_step)])]
                for i in range(len(traj))]
        else:
            double_rtn = [
                [rtn_1[i, np.argmax(rtn_2[i, :min(i + 1, self.max_step)])],
                 rtn_2[i, np.argmax(rtn_1[i, :min(i + 1, self.max_step)])]] for i
                in
                range(len(traj))]
        one_step_q = np.array([rtn_1[:, 0], rtn_2[:, 0]]).transpose()
        self.q_values[traj] = np.maximum(np.array(double_rtn), one_step_q)

    def update_memory(self, q_base=0, use_knn=False, beta=-1):
        # discount_beta = beta ** np.arange(self.max_step)
        trajs = self.retrieve_trajectories()
        self.compute_approximate_return_double_batch(batch_size=self.batch_size)
        for traj in trajs:
            self.update_traj(traj, q_base, beta)

        #     t = threading.Thread(target=self.update_traj, args=(traj, q_base,beta))
        #     threads.append(t)
        #     t.start()
        # for t in threads:
        #     t.join()
