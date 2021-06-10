import numpy as np
import copy
from envs import make_env, envs_collection
from utils.os_utils import get_arg_parser, get_logger, str2bool
from algorithm import create_agent, get_policy_train_type
from learner import create_learner, learner_collection
from test import Tester
from algorithm import algorithm_collection
from algorithm.replay_buffer import create_buffer, buffer_collection


def get_args():
    parser = get_arg_parser()

    # basic arguments
    parser.add_argument('--tag', help='terminal tag in logger', type=str, default='')
    parser.add_argument('--gpu', help='which gpu to use', type=int, default=0)
    parser.add_argument('--env', help='gym env id', type=str, default='Pong')
    parser.add_argument('--alg', help='backend algorithm', type=str, default='dqn', choices=algorithm_collection.keys())
    parser.add_argument('--learn', help='type of training method', type=str, default='normal',
                        choices=learner_collection.keys())

    args, _ = parser.parse_known_args()

    # env arguments
    parser.add_argument('--gamma', help='discount factor', type=np.float32, default=0.99)

    def atari_args():
        parser.set_defaults(learn='atari')
        parser.add_argument('--sticky', help='whether to use sticky actions', type=str2bool, default=False)
        parser.add_argument('--xian', help='whether to use xian group', type=str2bool, default=False)
        parser.add_argument('--noop', help='number of noop actions while starting new episode', type=np.int32,
                            default=30)
        parser.add_argument('--frames', help='number of stacked frames', type=np.int32, default=4)
        parser.add_argument('--rews_scale', help='scale of rewards', type=np.float32, default=1.0)
        parser.add_argument('--test_eps', help='random action noise in atari testing', type=np.float32, default=0.001)

    env_args_collection = {
        'atari': atari_args
    }
    env_args_collection[envs_collection[args.env]]()

    # training arguments
    parser.add_argument('--epoches', help='number of epoches', type=np.int32, default=20)
    parser.add_argument('--cycles', help='number of cycles per epoch', type=np.int32, default=20)
    parser.add_argument('--iterations', help='number of iterations per cycle', type=np.int32, default=100)
    parser.add_argument('--timesteps', help='number of timesteps per iteration', type=np.int32, default=500)


    # testing arguments
    parser.add_argument('--test_rollouts', help='number of rollouts to test per cycle', type=np.int32, default=5)
    parser.add_argument('--test_timesteps', help='number of timesteps per rollout', type=np.int32, default=27000)
    parser.add_argument('--save_rews', help='save cumulative rewards', type=str2bool, default=False)
    parser.add_argument('--save_Q', help='save Q estimation', type=str2bool, default=False)

    # buffer arguments
    parser.add_argument('--buffer', help='type of replay buffer', type=str, default='default',
                        choices=buffer_collection)
    parser.add_argument('--buffer_size', help='number of transitions in replay buffer', type=np.int32, default=1000000)
    parser.add_argument('--batch_size', help='size of sample batch', type=np.int32, default=32)
    parser.add_argument('--warmup', help='number of timesteps for buffer warmup', type=np.int32, default=2000)

    #### modifyed !!!!!!!!!!!!!!

    # algorithm arguments
    def q_learning_args():
        parser.add_argument('--train_batches', help='number of batches to train per iteration', type=np.int32,
                            default=25)
        parser.add_argument('--train_target', help='frequency of target network updating', type=np.int32, default=8000)

        parser.add_argument('--eps_l', help='beginning percentage of epsilon greedy explorarion', type=np.float32,
                            default=1.00)
        parser.add_argument('--eps_r', help='final percentage of epsilon greedy explorarion', type=np.float32,
                            default=0.01)
        parser.add_argument('--eps_decay', help='number of steps to decay epsilon', type=np.int32, default=250000)

        parser.add_argument('--optimizer', help='optimizer to use', type=str, default='adam',
                            choices=['adam', 'rmsprop'])
        args, _ = parser.parse_known_args()
        if args.optimizer == 'adam':
            parser.add_argument('--q_lr', help='learning rate of value network', type=np.float32, default=0.625e-4)
            parser.add_argument('--Adam_eps', help='epsilon factor of Adam optimizer', type=np.float32, default=1.5e-4)
        elif args.optimizer == 'rmsprop':
            parser.add_argument('--q_lr', help='learning rate of value network', type=np.float32, default=2.5e-4)
            parser.add_argument('--RMSProp_decay', help='decay factor of RMSProp optimizer', type=np.float32,
                                default=0.95)
            parser.add_argument('--RMSProp_eps', help='epsilon factor of RMSProp optimizer', type=np.float32,
                                default=1e-2)

        parser.add_argument('--nstep', help='parameter for n-step bootstrapping', type=np.int32, default=1)

    def dqn_args():
        # q_learning_args()
        ddq_args()
        parser.add_argument('--double', help='whether to use double trick', type=str2bool, default=False)
        # parser.add_argument('--dueling', help='whether to use dueling trick', type=str2bool, default=False)

    def cddqn_args():
        q_learning_args()
        parser.add_argument('--dueling', help='whether to use dueling trick', type=str2bool, default=False)

    def mmdqn_args():
        q_learning_args()
        parser.add_argument('--dueling', help='whether to use dueling trick', type=str2bool, default=False)

    def lrdqn_args():
        q_learning_args()
        parser.add_argument('--double', help='whether to use double trick', type=str2bool, default=False)
        parser.add_argument('--rank', help='rank of value matrix', type=np.int32, default=3)
        parser.add_argument('--beta', help='weight of sparsity loss', type=np.float32, default=1.0)

    def ddq_args():
        q_learning_args()
        parser.add_argument('--inner_q_type',
                            help='whether to use td3 trick/double trick TD3:min double-Q:double none:mean', type=str,
                            default='min')
        # parser.add_argument('--td4', help='whether to use td3 trick ', type=str2bool, default=False)
        parser.add_argument('--alpha', help='leaky relu parameter', type=np.float, default=1.)
        parser.add_argument('--tau', help='parameter for smooth target update', type=np.float, default=1.)
        parser.add_argument('--num_q', help='number of q to use', type=np.int32, default=4)
        parser.add_argument('--beta', help='if >0 use lambda return, else use max', type=np.float32, default=-1.)
        parser.add_argument('--state_dim', help='for representation, no use now', type=np.int, default=32)
        parser.add_argument('--dueling', help='whether to use dueling trick', type=str2bool, default=False)
        parser.add_argument('--max_step', help='max step to truncate', type=int, default=-1)

    algorithm_args_collection = {
        'dqn': dqn_args,
        'cddqn': cddqn_args,
        'mmdqn': mmdqn_args,
        'avedqn': dqn_args,
        'lrdqn': lrdqn_args,
        'ddq': ddq_args,
        'ddq6': ddq_args,
        'amc': ddq_args,
    }
    algorithm_args_collection[args.alg]()

    # learner arguments
    def lb_args():
        parser.add_argument('--lb_type', help='type of lower-bound objective', type=str, default='hard',
                            choices=['hard', 'soft'])

    def hash_args():
        lb_args()
        parser.add_argument('--avg_n', help='number of trajectories for moving average', type=np.int32, default=5)

    learner_args_collection = {
        'atari_lb': lb_args,
        'atari_hash_lb': hash_args,
        'atari_vi_lb': hash_args
    }
    if args.learn in learner_args_collection.keys():
        learner_args_collection[args.learn]()

    args = parser.parse_args()
    get_policy_train_type(args)

    logger_name = args.alg + '-' + args.env + '-' + args.learn
    if args.tag != '': logger_name = args.tag + '-' + logger_name
    args.logger = get_logger(logger_name)

    for key, value in args.__dict__.items():
        if key != 'logger':
            args.logger.info('{}: {}'.format(key, value))

    return args


def experiment_setup(args):
    env = make_env(args)
    args.acts_dims = env.acts_dims
    args.obs_dims = env.obs_dims

    args.buffer = buffer = create_buffer(args)
    args.agent = agent = create_agent(args)
    args.agent_graph = agent.graph
    args.learner = learner = create_learner(args)
    args.logger.info('*** network initialization complete ***')
    args.tester = tester = Tester(args)
    args.logger.info('*** tester initialization complete ***')

    return env, agent, buffer, learner, tester
