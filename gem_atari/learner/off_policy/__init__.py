from .normal import NormalLearner
from .atari import AtariLearner
from .atari_lb import AtariLowerBoundedLearner
from .atari_hash_lb import AtariHashLowerBoundedLearner
from .atari_vi_lb import AtariValueIterationLowerBoundedLearner
from .atari_ddq import AtariDDQLearner

learner_collection = {
	'normal': NormalLearner,
	'atari': AtariLearner,
	'atari_lb': AtariLowerBoundedLearner,
	'atari_hash_lb': AtariHashLowerBoundedLearner,
	'atari_vi_lb': AtariValueIterationLowerBoundedLearner,
	'atari_ddq': AtariDDQLearner,
}

def create_learner(args):
	return learner_collection[args.learn](args)
