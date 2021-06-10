import tensorflow as tf
import numpy as np
from stable_baselines.common import tf_util, SetVerbosity
from stable_baselines.common.buffers import ReplayBuffer
from stable_baselines.td3.episodic_memory import EpisodicMemory
from stable_baselines.td3.episodic_memory_tbp import EpisodicMemoryTBP
from stable_baselines.td3.td3_mem_backprop import TD3MemBackProp


class TD3MemGEM(TD3MemBackProp):
    """
    Twin Delayed DDPG (TD3)
    Addressing Function Approximation Error in Actor-Critic Methods.

    Original implementation: https://github.com/sfujim/TD3
    Paper: https://arxiv.org/pdf/1802.09477.pdf
    Introduction to TD3: https://spinningup.openai.com/en/latest/algorithms/td3.html

    :param policy: (TD3Policy or str) The policy model to use (MlpPolicy, CnnPolicy, LnMlpPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) the discount factor
    :param learning_rate: (float or callable) learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values and Actor networks)
        it can be a function of the current progress (from 1 to 0)
    :param buffer_size: (int) size of the replay buffer
    :param batch_size: (int) Minibatch size for each gradient update
    :param tau: (float) the soft update coefficient ("polyak update" of the target networks, between 0 and 1)
    :param policy_delay: (int) Policy and target networks will only be updated once every policy_delay steps
        per training steps. The Q values will be updated policy_delay more often (update every training step).
    :param action_noise: (ActionNoise) the action noise type. Cf DDPG for the different action noise type.
    :param target_policy_noise: (float) Standard deviation of Gaussian noise added to target policy
        (smoothing noise)
    :param target_noise_clip: (float) Limit for absolute value of target policy smoothing noise.
    :param train_freq: (int) Update the model every `train_freq` steps.
    :param learning_starts: (int) how many steps of the model to collect transitions for before learning starts
    :param gradient_steps: (int) How many gradient update after each step
    :param random_exploration: (float) Probability of taking a random action (as in an epsilon-greedy strategy)
        This is not needed for TD3 normally but can help exploring when using HER + TD3.
        This hack was present in the original OpenAI Baselines repo (DDPG + HER)
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        Note: this has no effect on TD3 logging for now
    :param seed: (int) Seed for the pseudo-random generators (python, numpy, tensorflow).
        If None (default), use random seed. Note that if you want completely deterministic
        results, you must set `n_cpu_tf_sess` to 1.
    :param n_cpu_tf_sess: (int) The number of threads for TensorFlow operations
        If None, the number of cpu of the current machine will be used.
    """

    def __init__(self, policy, env, eval_env, gamma=0.99, learning_rate=3e-4,
                 buffer_size=50000,
                 learning_starts=100, train_freq=100, gradient_steps=100, batch_size=128,
                 tau=0.005, policy_delay=1, qvalue_delay=1, action_noise=None, max_step=1000,
                 nb_eval_steps=1000, alpha=0.5, beta=-1, num_q=4, iterative_q=True, reward_scale=1.,
                 target_policy_noise=0.2, target_noise_clip=0.5, start_policy_learning=0,
                 random_exploration=0.0, verbose=0, tensorboard_log=None,
                 _init_setup_model=True, policy_kwargs=None, double_type="identical",
                 full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None):
        print("GEM Agent Here")

        # if iterative_q:
        #     gradient_steps *= 2
        #     policy_delay *= 2

        self.train_values_op_1 = None
        self.train_values_op_2 = None
        self.qf1_loss = None
        self.qf2_loss = None
        self.step_ops_1 = None
        self.step_ops_2 = None
        self.qfs_loss = None
        self.num_q = num_q  # or 2
        self.double_type = double_type
        self.clip_norm = 1
        super(TD3MemGEM, self).__init__(policy, env, eval_env, gamma, learning_rate,
                                        buffer_size,
                                        learning_starts, train_freq, gradient_steps, batch_size,
                                        tau, policy_delay, qvalue_delay, max_step, action_noise,
                                        nb_eval_steps, alpha, beta, num_q, iterative_q, reward_scale,
                                        target_policy_noise, target_noise_clip, start_policy_learning,
                                        random_exploration, verbose, tensorboard_log,
                                        _init_setup_model, policy_kwargs,
                                        full_tensorboard_log, seed, n_cpu_tf_sess)

    def setup_model(self):
        # print("setup model ",self.observation_space.shape)
        with SetVerbosity(self.verbose):
            self.graph = tf.Graph()
            with self.graph.as_default():
                self.set_random_seed(self.seed)
                self.sess = tf_util.make_session(num_cpu=self.n_cpu_tf_sess, graph=self.graph)

                self.replay_buffer = ReplayBuffer(self.buffer_size)

                with tf.variable_scope("input", reuse=False):
                    # Create policy and target TF objects
                    self.policy_tf = self.policy(self.sess, self.observation_space, self.action_space,
                                                 **self.policy_kwargs)
                    self.target_policy_tf = self.policy(self.sess, self.observation_space, self.action_space,
                                                        **self.policy_kwargs)

                    # Initialize Placeholders
                    self.observations_ph = self.policy_tf.obs_ph
                    # Normalized observation for pixels
                    self.processed_obs_ph = self.policy_tf.processed_obs
                    self.next_observations_ph = self.target_policy_tf.obs_ph
                    self.processed_next_obs_ph = self.target_policy_tf.processed_obs
                    self.action_target = self.target_policy_tf.action_ph
                    self.terminals_ph = tf.placeholder(tf.float32, shape=(None, 1), name='terminals')
                    self.rewards_ph = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
                    self.actions_ph = tf.placeholder(tf.float32, shape=(None,) + self.action_space.shape,
                                                     name='actions')
                    self.qvalues_ph = tf.placeholder(tf.float32,
                                                     shape=(None, self.num_q),
                                                     name='qvalues')
                    self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")

                with tf.variable_scope("model", reuse=False):
                    # Create the policy
                    self.policy_out = policy_out = self.policy_tf.make_actor(self.processed_obs_ph)
                    # Use two Q-functions to improve performance by reducing overestimation bias
                    qfs = self.policy_tf.make_many_critics(self.processed_obs_ph, self.actions_ph,
                                                           scope="buffer_values_fn", num_q=self.num_q)
                    # Q value when following the current policy
                    self.qfs = qfs
                    self.qfs_pi = self.policy_tf.make_many_critics(self.processed_obs_ph,
                                                                   policy_out, scope="buffer_values_fn",
                                                                   num_q=self.num_q, reuse=True)

                with tf.variable_scope("target", reuse=False):
                    # Create target networks
                    target_policy_out = self.target_policy_tf.make_actor(self.processed_next_obs_ph)

                    # Target policy smoothing, by adding clipped noise to target actions
                    target_noise = tf.random_normal(tf.shape(target_policy_out), stddev=self.target_policy_noise)
                    target_noise = tf.clip_by_value(target_noise, -self.target_noise_clip, self.target_noise_clip)
                    # Clip the noisy action to remain in the bounds [-1, 1] (output of a tanh)
                    noisy_target_action = tf.clip_by_value(target_policy_out + target_noise, -1, 1)

                    # Q values when following the target policy
                    qfs_target = self.target_policy_tf.make_many_critics(self.processed_next_obs_ph,
                                                                         # target_policy_out,
                                                                         noisy_target_action,
                                                                         scope="buffer_values_fn",
                                                                         num_q=self.num_q,
                                                                         reuse=False)

                    self.qfs_target = qfs_target
                    self.qfs_target_no_pi = self.target_policy_tf.make_many_critics(
                        self.processed_obs_ph,
                        self.actions_ph,
                        scope="buffer_values_fn", num_q=self.num_q, reuse=True)

                with tf.variable_scope("loss", reuse=False):
                    # Take the min of the two target Q-Values (clipped Double-Q Learning)
                    min_qf_target = tf.reduce_mean(qfs_target, axis=0) - self.q_base
                    # min_qf_target = tf.minimum(qf1_target, qf2_target)
                    print("here", min_qf_target.shape)
                    # Targets for Q value regression
                    q_backup = tf.stop_gradient(
                        self.rewards_ph +
                        (1 - self.terminals_ph) * self.gamma * min_qf_target
                    )
                    self.q_backup = q_backup
                    # Compute Q-Function loss

                    # Method 2
                    alpha = self.alpha
                    if alpha > 1:
                        alpha = 1. / alpha
                        sign = 1
                    else:
                        sign = -1

                    if self.double_type == "inner":
                        qfs = tf.reshape(qfs, (self.num_q, self.batch_size, 2))
                        qfs = tf.transpose(qfs, [1, 2, 0])
                        qfs = tf.reshape(qfs, (self.batch_size, 2, self.num_q // 2, 2))
                        qfs = tf.stack([qfs[:, 0, :, 0], qfs[:, 1, :, 1]], axis=-1)
                        qfs = tf.reshape(qfs, (self.batch_size, self.num_q))
                    elif self.double_type == "both":
                        qfs = tf.reshape(qfs, (self.num_q, self.batch_size, self.num_q))
                        qfs = tf.transpose(qfs, [1, 2, 0])
                        qfs = tf.stack([qfs[:, i, i] for i in range(self.num_q)], axis=-1)
                        qfs = tf.reshape(qfs, (self.batch_size, self.num_q))

                    diff = self.qvalues_ph - qfs + self.q_base
                    qfs_loss = tf.reduce_mean(
                        tf.nn.leaky_relu(sign * diff, alpha=alpha) ** 2) / alpha
                    self.qfs_loss = qfs_loss

                    qf1 = self.qfs[0, :]
                    qf2 = self.qfs[1, :]

                    qf1_loss = tf.reduce_mean(
                        tf.nn.leaky_relu(sign * (self.qvalues_ph - qf1 + self.q_base), alpha=alpha) ** 2) / alpha
                    qf2_loss = tf.reduce_mean(
                        tf.nn.leaky_relu(sign * (self.qvalues_ph - qf2 + self.q_base), alpha=alpha) ** 2) / alpha

                    qvalues_losses = qfs_loss

                    self.policy_loss = policy_loss = -tf.reduce_mean(self.qfs_pi)

                    # Policy loss: maximise q value

                    # Policy train op
                    # will be called only every n training steps,
                    # where n is the policy delay
                    policy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                    policy_train_op = policy_optimizer.minimize(policy_loss,
                                                                var_list=tf_util.get_trainable_vars('model/pi'))
                    self.policy_train_op = policy_train_op

                    # Q Values optimizer
                    qvalues_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                    qvalues_params = tf_util.get_trainable_vars('model/values_fn/') + tf_util.get_trainable_vars(
                        'model/buffer_values_fn/')

                    # Q Values and policy target params
                    source_params = tf_util.get_trainable_vars("model/")
                    target_params = tf_util.get_trainable_vars("target/")

                    # Polyak averaging for target variables
                    # self.target_ops = [
                    #     tf.assign(target, (1 - self.tau) * target + self.tau * source)
                    #     for target, source in zip(target_params, source_params)
                    # ]
                    self.target_ops = [
                        tf.assign(target,
                                  (1 - self.tau) ** (self.gradient_steps * 1) * target +
                                  (1 - (1 - self.tau) ** (self.gradient_steps * 1)) * source)
                        for target, source in zip(target_params, source_params)
                    ]
                    # self.target_ops = [
                    #     tf.assign(target, source)
                    #     for target, source in zip(target_params, source_params)
                    # ]
                    # Initializing target to match source variables
                    target_init_op = [
                        tf.assign(target, source)
                        for target, source in zip(target_params, source_params)
                    ]

                    grads = tf.gradients(qvalues_losses, qvalues_params)
                    grad_norm = tf.linalg.global_norm(grads)
                    # if self.clip_norm is not None:
                    #     grads = [tf.clip_by_norm(grad, clip_norm=self.clip_norm) for grad in grads]
                    # train_values_op = qvalues_optimizer.apply_gradients(grads)
                    train_values_op = qvalues_optimizer.minimize(qvalues_losses, var_list=qvalues_params)
                    # self.train_values_op_1 = qvalues_optimizer.minimize(qf1_loss, var_list=qvalues_params)
                    # self.train_values_op_2 = qvalues_optimizer.minimize(qf2_loss, var_list=qvalues_params)

                    self.infos_names = ['qfs_loss', 'q_grad_norm']
                    # All ops to call during one training step
                    self.step_ops = [qfs_loss, grad_norm,
                                     qfs, train_values_op]

                    self.step_ops_1 = [qf1_loss, qf1, self.train_values_op_1]
                    self.step_ops_2 = [qf2_loss, qf2, self.train_values_op_2]
                    # Monitor losses and entropy in tensorboard
                    tf.summary.scalar('policy_loss', policy_loss)
                    tf.summary.scalar('qfs_loss', qfs_loss)
                    tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate_ph))

                # Retrieve parameters that must be saved
                self.params = tf_util.get_trainable_vars("model")
                self.target_params = tf_util.get_trainable_vars("target/")

                # Initialize Variables and target network
                with self.sess.as_default():
                    self.sess.run(tf.global_variables_initializer())
                    self.sess.run(target_init_op)

                self.summary = tf.summary.merge_all()

                self.memory = EpisodicMemoryTBP(self.buffer_size, state_dim=1,
                                                obs_space=self.observation_space,
                                                action_shape=self.action_space.shape,
                                                q_func=self.qfs_target, repr_func=None,
                                                obs_ph=self.processed_next_obs_ph,
                                                action_ph=self.actions_ph, sess=self.sess, gamma=self.gamma,
                                                max_step=self.max_step)

    def _train_step(self, step, writer, learning_rate, update_policy, update_q):
        # Sample a batch from the replay buffer
        # batch = self.replay_buffer.sample(self.batch_size, env=self._vec_normalize_env)
        # cur_time = time.time()
        num_samples_collection = {"identical": self.batch_size, "inner": self.batch_size * self.num_q // 2,
                                  "both": self.batch_size * self.num_q}
        num_samples = num_samples_collection[self.double_type]
        # num_samples = self.batch_size * self.num_q//2 if self.iterative_q else self.batch_size
        batch = self.memory.sample(num_samples, mix=False)
        if batch is None:
            return 0
        batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones, batch_returns = batch['obs0'], batch[
            'actions'], batch['rewards'], batch['obs1'], batch['terminals1'], batch['return']

        if self.double_type == "identical":
            batch_returns = np.repeat(batch_returns, self.num_q // 2, axis=1)
        else:
            if self.double_type == "both":
                batch_returns_1 = batch_returns[:self.batch_size * self.num_q // 2, :1]
                batch_returns_2 = batch_returns[self.batch_size * self.num_q // 2:, 1:]
                batch_returns = np.concatenate([batch_returns_1, batch_returns_2], axis=1)
            batch_returns = np.reshape(batch_returns, (self.batch_size, self.num_q // 2, 2))
            batch_returns = np.swapaxes(batch_returns, 1, -1)
            batch_returns = np.reshape(batch_returns, (self.batch_size, self.num_q))
        # if self.iterative_q:
        #     batch_returns = batch_returns[:, :self.num_q // 2] if step % 2 == 0 else batch_returns[:, self.num_q // 2:]
        # print("Sample time: ",time.time()-cur_time)
        # cur_time = time.time()
        feed_dict = {
            self.observations_ph: batch_obs,
            self.actions_ph: batch_actions,
            self.next_observations_ph: batch_next_obs,
            self.rewards_ph: batch_rewards.reshape(num_samples, -1),
            self.terminals_ph: batch_dones.reshape(num_samples, -1),
            self.learning_rate_ph: learning_rate,
            self.qvalues_ph: batch_returns
        }
        # print("training ",batch_obs.shape)
        if update_q:
            step_ops = self.step_ops
            # if not self.iterative_q:
            #     step_ops = self.step_ops
            # else:
            #     step_ops = self.step_ops_1 if step % 2 == 0 else self.step_ops_2
        else:
            step_ops = [self.qfs_loss]

        if update_policy:
            # Update policy and target networks
            step_ops = step_ops + [self.policy_train_op, self.policy_loss]

        if not step_ops:
            return 0  # not updating q nor policy
        # Do one gradient step
        # and optionally compute log for tensorboard
        if writer is not None:
            out = self.sess.run([self.summary] + step_ops, feed_dict)
            summary = out.pop(0)
            writer.add_summary(summary, step)
        else:
            out = self.sess.run(step_ops, feed_dict)

        # Unpack to monitor losses
        qfs_loss, grad_norm, *_values = out

        return qfs_loss, grad_norm
