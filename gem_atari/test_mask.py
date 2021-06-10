from utils.os_utils import get_arg_parser, str2bool
from envs import make_env
import numpy as np
import cv2
import matplotlib.pyplot as plt
from algorithm.hash import Hash

parser = get_arg_parser()
parser.add_argument('--env', help='gym env id', type=str, default='Alien')
parser.add_argument('--sticky', help='whether to use sticky actions', type=str2bool, default=False)
parser.add_argument('--noop', help='number of noop actions while starting new episode', type=np.int32, default=30)
parser.add_argument('--frames', help='number of stacked frames', type=np.int32, default=4)
parser.add_argument('--test_timesteps', help='number of timesteps per rollout', type=np.int32, default=1000)
parser.add_argument('--avg_n', help='number of trajectories for moving average', type=np.int32, default=5)
args = parser.parse_args()

hash_lib = Hash(args)

env = make_env(args)
env.env.seed(233)
obs = env.reset()
for i in range(200):
    obs, rew, done, _ = env.step(env.action_space.sample())
    env.render()
    if done: break

print(obs.shape)
plt.imshow(hash_lib.score_mask(obs[:,:,-1]))
plt.show()
