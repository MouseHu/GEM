import gym
import numpy as np
import time
from stable_baselines.ddpg.noise import AdaptiveParamNoiseSpec, OrnsteinUhlenbeckActionNoise, NormalActionNoise
from stable_baselines.common.wrappers import TimestepWrapper, DelayedRewardWrapper, NHWCWrapper
# from toy_env import *
# import dmc2gym
from stable_baselines import logger, bench
from stable_baselines.common.misc_util import set_global_seeds, boolean_flag
import argparse
import os
import json


def create_env(env_id, delay_step, env_str=str(0)):
    # if env_type in ["mujoco", "Mujoco", "MuJoCo", "raw", "mujoco_raw", "raw_mujoco"]:
    env = gym.make(env_id)
    env = TimestepWrapper(env)
    env = DelayedRewardWrapper(env, delay_step)
    env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), env_str))
    return env


def create_action_noise(env, noise_type):
    action_noise = None
    nb_actions = env.action_space.shape[-1]
    for current_noise_type in noise_type.split(','):
        current_noise_type = current_noise_type.strip()
        if current_noise_type == 'none':
            pass
        elif 'adaptive-param' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
        elif 'normal' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = NormalActionNoise(mean=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
            # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        elif 'ou' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(nb_actions),
                                                        sigma=float(stddev) * np.ones(nb_actions))
        else:
            raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))
    return action_noise


def save_args(args):
    log_dir = os.getenv("OPENAI_LOGDIR")
    os.makedirs(log_dir, exist_ok=True)
    param_file = os.path.join(log_dir, "params.txt")
    with open(param_file, "w") as pf:
        pf.write(json.dumps(args))


def parse_args():
    """
    parse the arguments for DDPG training

    :return: (dict) the arguments
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env_type', type=str, default="mujoco")
    parser.add_argument('--env-id', type=str, default='Ant-v2')
    parser.add_argument('--agent', type=str, default='TD3')
    # boolean_flag(parser, 'render-eval', default=False)
    boolean_flag(parser, 'layer-norm', default=False)
    boolean_flag(parser, 'evaluation', default=True)

    parser.add_argument('--seed', help='RNG seed', type=int, default=int(time.time()))
    parser.add_argument('--comment', help='to show name', type=str, default="show_name_in_htop")

    parser.add_argument('--gamma', type=float, default=0.99)

    parser.add_argument('--num-timesteps', type=int, default=int(1e6)+10)  # plus 10 to make one more evaluation
    parser.add_argument('--max_steps', type=int, default=1000)  # truncate steps for ddq

    parser.add_argument('--delay-step', type=int, default=0)

    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args
