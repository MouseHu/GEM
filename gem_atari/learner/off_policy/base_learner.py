import tensorflow as tf
# import horovod.tensorflow as hvd

class BaseLearner(object):
    def __init__(self):
        self.q_loss = None
        self.meta_q_loss = None
        self.q_l1_loss = None
        self.q_diff = None
        self.target_check_range = None

        self.q_pi = None

        self.q_t = None

        self.acts_ph = None
        self.obs_next_ph = None
        self.obs_ph = None

        self.raw_obs_ph = None
        self.raw_obs_next_ph = None
        self.em_raw_obs_ph = None
        self.em_obs_ph = None
        self.q_funcs_stack = None

        self.rews_ph = None
        self.done_ph = None

        self.q_lb_ph = None
        self.q_step = None
        self.q_step_target = None

        self.sess = None

        self.init_op = None
        self.target_init_op = None

        self.q_train_op = None
        self.meta_q_train_op = None
        self.obs_normalizer = None
        self.target_update_op = None
        self.qvalues_ph = None
        self.acts_num = 0
        self.args = None
        self.graph = None
        self.em_q= None
        self.true_rews_ph = None

        self.q = None

        self.buffer_target_diffence = None
        self.q_target_mean = None


    def mlp_value(self, obs_ph):
        with tf.variable_scope('net', initializer=tf.contrib.layers.xavier_initializer()):
            q_dense1 = tf.layers.dense(obs_ph, 256, activation=tf.nn.relu, name='q_dense1')
            q_dense2 = tf.layers.dense(q_dense1, 256, activation=tf.nn.relu, name='q_dense2')
            q = tf.layers.dense(q_dense2, self.acts_num, name='q')
        return q

    def conv_value(self, obs_ph):
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

    def create_session(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
        # config = tf.ConfigProto(device_count={'GPU': 1}, gpu_options=gpu_options)
        config = tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True,log_device_placement=True)
        config.gpu_options.visible_device_list = str(self.args.gpu)
        # if self.args.xian:
        #     config.gpu_options.visible_device_list = str(hvd.local_rank())
        # config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)