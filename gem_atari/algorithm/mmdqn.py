import numpy as np
import tensorflow as tf
from utils.tf_utils import get_vars, Normalizer

class MaxminDQN:
	def __init__(self, args, flags={}):
		self.args = args
		self.flags = flags
		self.acts_num = args.acts_dims[0]
		self.create_model()

		self.train_info = {
			'Q_loss': self.q_loss,
			'Q_L1_loss': self.q_l1_loss,
			'Q_diff': self.q_diff,
			'target_range': self.target_check_range
		}
		self.step_info = {
			'Q_average': self.q_pi
		}

		if self.args.learn[-2:]=='lb':
			self.train_info = {
				**self.train_info,
				**{
					'Q_target': self.target,
					'Q_LB': self.q_lb_ph,
					'LB_ratio': self.lb_ratio
				}
			}

	def create_model(self):
		def create_session():
			config = tf.ConfigProto()
			config.gpu_options.allow_growth = True
			self.sess = tf.Session(config=config)

		def create_inputs():
			self.raw_obs_ph = tf.placeholder(tf.float32, [None]+self.args.obs_dims)
			self.raw_obs_next_ph = tf.placeholder(tf.float32, [None]+self.args.obs_dims)
			self.acts_ph = tf.placeholder(tf.float32, [None]+self.args.acts_dims)
			self.rews_ph = tf.placeholder(tf.float32, [None, 1])
			self.done_ph = tf.placeholder(tf.float32, [None, 1])

			if self.args.learn[-2:]=='lb':
				self.q_lb_ph = tf.placeholder(tf.float32, [None, 1])

		def create_normalizer():
			if len(self.args.obs_dims)==1:
				with tf.variable_scope('normalizer'):
					self.obs_normalizer = Normalizer(self.args.obs_dims, self.sess)
				self.obs_ph = self.obs_normalizer.normalize(self.raw_obs_ph)
				self.obs_next_ph = self.obs_normalizer.normalize(self.raw_obs_next_ph)
			else:
				self.obs_normalizer = None
				self.obs_ph = self.raw_obs_ph
				self.obs_next_ph = self.raw_obs_next_ph

		def create_network():
			def mlp_value(obs_ph):
				with tf.variable_scope('net', initializer=tf.contrib.layers.xavier_initializer()):
					q_dense1 = tf.layers.dense(obs_ph, 256, activation=tf.nn.relu, name='q_dense1')
					q_dense2 = tf.layers.dense(q_dense1, 256, activation=tf.nn.relu, name='q_dense2')
					q = tf.layers.dense(q_dense2, self.acts_num, name='q')
				return q

			def conv_value(obs_ph):
				with tf.variable_scope('net', initializer=tf.contrib.layers.xavier_initializer()):
					q_conv1 = tf.layers.conv2d(obs_ph, 32, 8, 4, 'same', activation=tf.nn.relu, name='q_conv1')
					q_conv2 = tf.layers.conv2d(q_conv1, 64, 4, 2, 'same', activation=tf.nn.relu, name='q_conv2')
					q_conv3 = tf.layers.conv2d(q_conv2, 64, 3, 1, 'same', activation=tf.nn.relu, name='q_conv3')
					q_conv3_flat = tf.layers.flatten(q_conv3)

					q_dense_act = tf.layers.dense(q_conv3_flat, 512, activation=tf.nn.relu, name='q_dense_act')
					q_act = tf.layers.dense(q_dense_act, self.acts_num, name='q_act')

					if self.args.dueling:
						q_dense_base = tf.layers.dense(q_conv3_flat, 512, activation=tf.nn.relu, name='q_dense_base')
						q_base = tf.layers.dense(q_dense_base, 1, name='q_base')
						q = q_base + q_act - tf.reduce_mean(q_act, axis=1, keepdims=True)
					else:
						q = q_act
				return q

			value_net = mlp_value if len(self.args.obs_dims)==1 else conv_value

			with tf.variable_scope('main'):
				with tf.variable_scope('value_1'):
					self.q = value_net(self.obs_ph)
					self.q_pi = tf.reduce_max(self.q, axis=1, keepdims=True)
				with tf.variable_scope('value_2'):
					self.q_2 = value_net(self.obs_ph)

			with tf.variable_scope('target'):
				with tf.variable_scope('value_1'):
					self.q_t_1 = value_net(self.obs_next_ph)
				with tf.variable_scope('value_2'):
					self.q_t_2 = value_net(self.obs_next_ph)
				self.q_t = tf.reduce_max(tf.minimum(self.q_t_1, self.q_t_2), axis=1, keepdims=True)

		def create_operators():
			self.target = tf.stop_gradient(self.rews_ph+(1.0-self.done_ph)*(self.args.gamma**self.args.nstep)*self.q_t)
			target = self.target
			if self.args.learn[-2:]=='lb':
				self.lb_ratio = tf.less(target, self.q_lb_ph)
				target = tf.maximum(target, self.q_lb_ph)
			self.target_check_range = tf.reduce_max(tf.abs(target))
			self.q_acts = tf.reduce_sum(self.q*self.acts_ph, axis=1, keepdims=True)
			self.q_acts_2 = tf.reduce_sum(self.q_2*self.acts_ph, axis=1, keepdims=True)
			self.q_loss = tf.losses.huber_loss(target, self.q_acts) + tf.losses.huber_loss(target, self.q_acts_2)
			self.q_diff = tf.reduce_mean(tf.abs(self.q_acts-self.q_acts_2))
			self.q_l1_loss = tf.reduce_mean(tf.abs(target-self.q_acts))
			if self.args.optimizer=='adam':
				self.q_optimizer = tf.train.AdamOptimizer(self.args.q_lr, epsilon=self.args.Adam_eps)
			elif self.args.optimizer=='rmsprop':
				self.q_optimizer = tf.train.RMSPropOptimizer(self.args.q_lr, decay=self.args.RMSProp_decay, epsilon=self.args.RMSProp_eps)
			self.q_train_op = self.q_optimizer.minimize(self.q_loss, var_list=get_vars('main/value'))

			self.target_update_op = tf.group([
				v_t.assign(v)
				for v, v_t in zip(get_vars('main'), get_vars('target'))
			])

			self.saver=tf.train.Saver()
			self.init_op = tf.global_variables_initializer()
			self.target_init_op = tf.group([
				v_t.assign(v)
				for v, v_t in zip(get_vars('main'), get_vars('target'))
			])

		self.graph = tf.Graph()
		with self.graph.as_default():
			create_session()
			create_inputs()
			create_normalizer()
			create_network()
			create_operators()
		self.init_network()

	def init_network(self):
		self.sess.run(self.init_op)
		self.sess.run(self.target_init_op)

	def step(self, obs, explore=False, test_info=False):
		if (not test_info) and (self.args.buffer.steps_counter<self.args.warmup):
			return np.random.randint(self.acts_num)

		# eps-greedy exploration
		if explore and np.random.uniform()<=self.args.eps_act:
			return np.random.randint(self.acts_num)

		feed_dict = {
			self.raw_obs_ph: [obs/255.0]
		}
		q_value, info = self.sess.run([self.q, self.step_info], feed_dict)
		action = np.argmax(q_value[0])

		if test_info: return action, info
		return action

	def feed_dict(self, batch):
		def one_hot(idx):
			idx = np.array(idx)
			batch_size = idx.shape[0]
			res = np.zeros((batch_size, self.acts_num), dtype=np.float32)
			res[np.arange(batch_size),idx] = 1.0
			return res

		feed_dict = {
			self.raw_obs_ph: np.array(batch['obs']),
			self.raw_obs_next_ph: np.array(batch['obs_next']),
			self.acts_ph: one_hot(batch['acts']),
			self.rews_ph: np.clip(np.array(batch['rews']), -self.args.rews_scale, self.args.rews_scale),
			self.done_ph: batch['done']
		}

		if self.args.learn[-2:]=='lb':
			feed_dict[self.q_lb_ph]= batch['rets']

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
		if not(self.obs_normalizer is None):
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
