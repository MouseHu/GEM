from algorithm.episodic_memory import EpisodicMemory
import numpy as np
import time


class EpisodicMemoryDDQ(EpisodicMemory):
    def __init__(self, args):

        super(EpisodicMemoryDDQ, self).__init__(args)
        del self._q_values
        self._q_values = -np.inf * np.ones((args.buffer_size + 1, 2))
        self.max_step = args.max_step if args.max_step>0 else 1e6
        self.inner_q_type = args.inner_q_type
        # self.batch_size = args.batch_size * 16

    def compute_approximate_return_double(self, obses, actions=None):
        returns = []
        num_epoch = int(np.ceil(len(obses)/self.batch_size))
        for e in range(num_epoch):
            low = e*self.batch_size
            high = min(len(obses),(e+1)*self.batch_size)
            batch_obs = obses[low:high].astype(np.float32) / 255.0
            returns.append(np.array(self.sess.run(self.q_func, feed_dict={self.obs_ph: batch_obs})).reshape(-1,2))
        return np.concatenate(returns)

    def update_memory(self, q_base=0, use_knn=False, beta=-1):
        discount_beta = beta ** np.arange(self.max_step)
        trajs = self.retrieve_trajectories()
        for traj in trajs:
            # print(np.array(traj))
            approximate_qs = self.compute_approximate_return_double(self.get_obs(traj), self.action_buffer[traj])
            approximate_qs = approximate_qs.transpose()
            if len(approximate_qs) >= 4:
                num_q = len(approximate_qs)
                approximate_qs = approximate_qs.reshape((2, num_q // 2, -1))
                if self.inner_q_type == "min":
                    approximate_qs = np.min(approximate_qs, axis=1)  # clip double q
                else:
                    approximate_qs = np.mean(approximate_qs, axis=1)  # clip double q
                # approximate_qs = approximate_qs.transpose()
            else:

                assert len(approximate_qs) == 2, approximate_qs.shape
                approximate_qs = approximate_qs.reshape(2, -1)
            approximate_qs = np.concatenate([np.zeros((2, 1)), approximate_qs], axis=1)

            self.q_values[traj] = 0
            rtn_1 = np.zeros((len(traj), min(self.max_step, len(traj))))
            rtn_2 = np.zeros((len(traj), min(self.max_step, len(traj))))

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

            # if self.inner_q_type == "min":
            #     lower_bound = np.min([rtn_1[:, 0], rtn_2[:, 0]], axis=0, keepdims=True).transpose()
            # else:
            #     lower_bound = np.mean([rtn_1[:, 0], rtn_2[:, 0]], axis=0, keepdims=True).transpose()
            lower_bound = np.array([rtn_1[:, 0], rtn_2[:, 0]]).transpose()
            self.q_values[traj] = np.maximum(np.array(double_rtn), lower_bound)
            # one_step_q = np.array([rtn_1[:, 0], rtn_2[:, 0]]).transpose()
            # self.q_values[traj] = np.maximum(np.array(double_rtn),
            #                                  np.min(one_step_q, axis=1, keepdims=True))

            # Rtd = np.mean(one_step_q, axis=1,keepdims=True)
            # self.q_values[traj] = np.repeat(Rtd, 2,axis=1)
            # one_step_q)

    def update_sequence_with_qs(self, sequence, beta=-1):
        # print(sequence)
        next_id = -1
        rtn_1 = np.zeros((len(sequence), min(len(sequence), self.max_step)))
        rtn_2 = np.zeros((len(sequence), min(len(sequence), self.max_step)))

        qs = [transition[3] for transition in reversed(sequence)]
        qs = [np.zeros(2)] + qs[:-1]

        for i, transition in enumerate(reversed(sequence)):
            obs, a, z, q_t, r, truly_done, done = transition
            r = np.clip(r, -self.rews_scale, self.rews_scale)
            rtn_1[i, 0], rtn_2[i, 0] = r + self.gamma * (1 - truly_done) * np.array(qs[i]).squeeze()

        episode_step = len(sequence)
        tRtd = 0
        for i, transition in enumerate(reversed(sequence)):
            episode_step -= 1
            obs, a, z, q_t, r, truly_done, done = transition
            r = np.clip(r, -self.rews_scale, self.rews_scale)
            tRtd = r + self.gamma * (1 - truly_done) * tRtd
            # print(np.mean(z))
            rtn_1[i, 1:] = r + self.gamma * rtn_1[i - 1, :-1]
            rtn_2[i, 1:] = r + self.gamma * rtn_2[i - 1, :-1]

            double_rtn = [rtn_1[i, np.argmax(rtn_2[i, :min(i + 1, self.max_step)])],
                          rtn_2[i, np.argmax(rtn_1[i, :min(i + 1, self.max_step)])]]

            # self.q_values[traj] = np.maximum(np.array(double_rtn),np.minimum(rtn_1[:,0],rtn_2[:,0]))
            one_step_q = np.array([rtn_1[i, 0], rtn_2[i, 0]]).transpose()

            # if self.inner_q_type == "min":
            #     lower_bound = np.min(one_step_q, keepdims=True)
            # else:
            #     lower_bound = np.mean(one_step_q, keepdims=True)
            lower_bound = one_step_q
            Rtd = np.maximum(np.array(double_rtn), lower_bound)
            # Rtd = np.mean(np.array(one_step_q), keepdims=True)
            # Rtd = np.repeat(Rtd,2,axis=0)
            current_id = self.add(obs, a, z, Rtd, next_id)

            if done:
                self.end_points.append(current_id)
            if episode_step == 0:
                self.begin_obs[current_id] = obs
            self.episode_steps[current_id] = episode_step
            self.frame_buffer[current_id] = obs[..., -1]
            self.reward_buffer[current_id] = r
            self.returns[current_id] = tRtd
            self.done_buffer[current_id] = done
            self.truly_done_buffer[current_id] = truly_done
            next_id = int(current_id)
        # self.update_priority()
        self.counter += 1
        return
