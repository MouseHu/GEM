import numpy as np
import tensorflow as tf
from utils.tf_utils import get_vars, Normalizer,huber_loss
from learner.off_policy.base_learner import BaseLearner
# import horovod.tensorflow as hvd


class DDQ(BaseLearner):
    def __init__(self, args, flags={}):
        super(DDQ, self).__init__()
        self.args = args
        self.gpu = args.gpu
        self.flags = flags
        self.acts_num = args.acts_dims[0]

        self.num_q = self.args.num_q
        self.tau = self.args.tau
        self.alpha = self.args.alpha
        self.beta = self.args.beta
        self.inner_q_type= args.inner_q_type

        self.q_funcs = []
        self.q_pi_funcs = []
        self.target_q_funcs = []
        self.target_q_pi_funcs = []
        self.create_model()

        self.train_info = {
            'Q_loss': self.q_loss,
            # 'Q_target_0': self.q_step_target[:, 0],
            # 'Q_target_1': self.q_step_target[:, 1],
            'difference': self.buffer_target_diffence,
            'target_range': self.target_check_range,
            'regression_target': self.qvalues_ph,
            'true_return': self.true_rews_ph,
        }
        self.step_info = {
            'Q_average': self.q_pi,

        }

        self.args.buffer.update_func(self)

    def create_model(self):
        # def create_session():
        #     config = tf.ConfigProto()
        #     config.gpu_options.allow_growth = True
        #     self.sess = tf.Session(config=config)

        def create_inputs():
            self.raw_obs_ph = tf.placeholder(tf.float32, [None] + self.args.obs_dims)
            self.em_raw_obs_ph = tf.placeholder(tf.float32, [None] + self.args.obs_dims)
            self.acts_ph = tf.placeholder(tf.float32, [None] + self.args.acts_dims + [1, 1])
            self.rews_ph = tf.placeholder(tf.float32, [None, 1])
            self.true_rews_ph = tf.placeholder(tf.float32, [None, 1])
            self.done_ph = tf.placeholder(tf.float32, [None, 1])
            self.qvalues_ph = tf.placeholder(tf.float32, [None, self.num_q])

        def create_normalizer():
            if len(self.args.obs_dims) == 1:
                with tf.variable_scope('normalizer'):
                    self.obs_normalizer = Normalizer(self.args.obs_dims, self.sess)
                self.obs_ph = self.obs_normalizer.normalize(self.raw_obs_ph)
                self.em_obs_ph = self.obs_normalizer.normalize(self.em_raw_obs_ph)
            else:
                self.obs_normalizer = None
                self.obs_ph = self.raw_obs_ph
                self.em_obs_ph = self.em_raw_obs_ph

        def create_network():

            value_net = self.mlp_value if len(self.args.obs_dims) == 1 else self.conv_value

            with tf.variable_scope('main'):
                for i in range(self.num_q):
                    with tf.variable_scope('value_{}'.format(i)):
                        q = value_net(self.obs_ph)
                        q_pi = tf.reduce_max(q, axis=1, keepdims=True)
                        self.q_funcs.append(q)
                        self.q_pi_funcs.append(q_pi)
                q_funcs_stack = tf.stack(self.q_funcs, axis=-1)
                self.q_funcs_stack = tf.reshape(q_funcs_stack, [-1] + self.args.acts_dims + [2, self.num_q // 2])
                if self.inner_q_type == "min":
                    self.q_step = tf.reduce_min(self.q_funcs_stack, axis=-1)
                else:
                    self.q_step = tf.reduce_mean(self.q_funcs_stack, axis=-1)
                self.q_pi = tf.reduce_mean(tf.reduce_max(self.q_step, axis=1), axis=-1)

            with tf.variable_scope('target'):
                for i in range(self.num_q):
                    with tf.variable_scope('value_{}'.format(i)):
                        tar_q = value_net(self.em_obs_ph)
                        tar_q_pi = tf.reduce_max(q, axis=1, keepdims=True)
                        self.target_q_funcs.append(tar_q)
                        self.target_q_pi_funcs.append(tar_q_pi)
                target_q_funcs_stack = tf.stack(self.target_q_funcs, axis=-1)
                self.target_q_funcs_stack = tf.reshape(target_q_funcs_stack,
                                                       [-1] + self.args.acts_dims + [2, self.num_q // 2])
                if self.inner_q_type == "min":
                    self.em_q = tf.reduce_max(tf.reduce_min(self.target_q_funcs_stack, axis=-1), axis=1)
                else:
                    self.em_q = tf.reduce_max(tf.reduce_mean(self.target_q_funcs_stack, axis=-1), axis=1)
                self.q_target_mean = tf.reduce_mean(self.em_q, axis=1, keepdims=True)
                # self.q_step_target = tf.reduce_mean(self.target_q_funcs_stack, axis=-1)

        def create_operators():

            self.target = tf.stop_gradient(
                self.rews_ph + (1.0 - self.done_ph) * self.args.gamma * self.q_target_mean)

            self.q_acts = tf.reduce_sum(self.q_funcs_stack * self.acts_ph, axis=1, keepdims=True)
            q_acts = tf.reshape(self.q_acts, shape=(-1, self.num_q))
            self.target_check_range = tf.reduce_max(tf.abs(self.qvalues_ph))
            self.buffer_target_diffence = tf.reduce_sum((self.qvalues_ph - self.target) ** 2)
            # self.q_loss = tf.reduce_mean(tf.abs(q_acts - self.qvalues_ph))
            # self.q_loss = tf.reduce_mean(tf.nn.leaky_relu(q_acts - self.qvalues_ph, alpha=self.alpha)**2)
            sym_q_loss = (q_acts-self.qvalues_ph)**2 # penalize overestimation of q near done
            # sym_q_loss = huber_loss(q_acts, self.qvalues_ph)
            overestimate = (q_acts - self.qvalues_ph) > 0
            self.q_loss = tf.reduce_mean(tf.where(overestimate, sym_q_loss, self.alpha * sym_q_loss))
            if self.args.optimizer == 'adam':
                self.q_optimizer = tf.train.AdamOptimizer(self.args.q_lr, epsilon=self.args.Adam_eps)
            elif self.args.optimizer == 'rmsprop':
                self.q_optimizer = tf.train.RMSPropOptimizer(self.args.q_lr, decay=self.args.RMSProp_decay,
                                                             epsilon=self.args.RMSProp_eps)
            self.q_train_op = self.q_optimizer.minimize(self.q_loss, var_list=get_vars('main'))

            self.target_update_op = tf.group([

                v_t.assign(self.tau * v + (1 - self.tau) * v_t)
                for v, v_t in zip(get_vars('main'), get_vars('target'))
            ])

            self.saver = tf.train.Saver()
            self.init_op = tf.global_variables_initializer()
            self.target_init_op = tf.group([
                v_t.assign(v)
                for v, v_t in zip(get_vars('main'), get_vars('target'))
            ])

        self.graph = tf.Graph()

        with self.graph.as_default():
            # with tf.device("/gpu:{}".format(self.gpu)):
            self.create_session()
            create_inputs()
            create_normalizer()
            create_network()
            create_operators()
        self.init_network()

    def init_network(self):
        self.sess.run(self.init_op)
        self.sess.run(self.target_init_op)

    def step(self, obs, explore=False, test_info=False):
        if (not test_info) and (self.args.buffer.steps_counter < self.args.warmup):
            return np.random.randint(self.acts_num)

        # eps-greedy exploration
        if explore and np.random.uniform() <= self.args.eps_act:
            return np.random.randint(self.acts_num)

        feed_dict = {
            self.raw_obs_ph: [obs / 255.0],
        }
        q_values, info = self.sess.run([self.q_step, self.step_info], feed_dict)
        q_value = np.mean(q_values, axis=-1)
        action = np.argmax(q_value[0])

        if test_info: return action, info
        return action

    def step_with_q(self, obs, explore=False, target=True):

        feed_dict = {
            self.raw_obs_ph: [obs / 255.0],
            self.em_raw_obs_ph: [obs / 255.0],
        }
        q_values, q_values_target = self.sess.run([self.q_step, self.em_q], feed_dict)
        q_value = np.mean(q_values, axis=-1)
        action = np.argmax(q_value[0])

        q = q_values_target if target else q_values
        if self.args.buffer.steps_counter < self.args.warmup:
            return np.random.randint(self.acts_num), q

        # eps-greedy exploration
        if explore and np.random.uniform() <= self.args.eps_act:
            return np.random.randint(self.acts_num), q

        return action, q

    def feed_dict(self, batch):
        def one_hot(idx):
            idx = np.array(idx).reshape(-1)
            batch_size = idx.shape[0]
            res = np.zeros((batch_size, self.acts_num), dtype=np.float32)
            res[np.arange(batch_size), idx] = 1.0
            return res

        batch_obs, batch_obs_next, batch_actions, batch_rewards, batch_next_obs, batch_dones, batch_returns, batch_true_returns = \
        batch['obs0'], batch['obs1'], batch[
            'actions'], batch['rewards'], batch['obs1'], batch['terminals1'], batch['return'], batch['true_return']
        if self.num_q == 4:
            batch_returns = np.repeat(batch_returns, 2, axis=1)
        feed_dict = {
            self.raw_obs_ph: batch_obs,
            self.em_raw_obs_ph: batch_obs_next,
            self.acts_ph: one_hot(batch_actions)[..., np.newaxis, np.newaxis],
            self.rews_ph: batch_rewards,
            self.done_ph: batch_dones,
            self.qvalues_ph: batch_returns,
            self.true_rews_ph: batch_true_returns,
        }

        return feed_dict

    def train(self, batch):
        feed_dict = self.feed_dict(batch)
        info, _ = self.sess.run([self.train_info, self.q_train_op], feed_dict)
        return info

    def test_q(self, batch):
        feed_dict = self.feed_dict(batch)
        q_loss = self.sess.run(self.q_loss, feed_dict)
        return q_loss

    def normalizer_update(self, batch):
        if not (self.obs_normalizer is None):
            self.obs_normalizer.update(np.concatenate([batch['obs'], batch['obs_next']], axis=0))

    def target_update(self):
        self.sess.run(self.target_update_op)

    def save_model(self, save_path):
        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.save(self.sess, save_path)

    def load_model(self, load_path):
        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.restore(self.sess, load_path)
