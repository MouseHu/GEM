from .dqn import DQN
from .cddqn import ClippedDDQN
from .mmdqn import MaxminDQN
from .avedqn import AveragedDQN
from .lrdqn import LowRankDQN
from .ddq import DDQ
from .ddq6 import DDQ6
from .amc import AMC

on_policy_tag = 'on-policy'
off_policy_tag = 'off-policy'

algorithm_collection = {
    # on-policy algorithm
    # none
    # off-policy algorithm
    'dqn': (DQN, off_policy_tag),
    'cddqn': (ClippedDDQN, off_policy_tag),
    'mmdqn': (MaxminDQN, off_policy_tag),
    'avedqn': (AveragedDQN, off_policy_tag),
    'lrdqn': (LowRankDQN, off_policy_tag),
    'ddq': (DDQ, off_policy_tag),
    'ddq6': (DDQ6, off_policy_tag),
    'amc': (AMC, off_policy_tag),
}


def create_agent(args):
    return algorithm_collection[args.alg][0](args)


def get_policy_train_type(args):
    policy_tag = algorithm_collection[args.alg][1]
    args.on_policy = (policy_tag == on_policy_tag)
    args.off_policy = (policy_tag == off_policy_tag)
