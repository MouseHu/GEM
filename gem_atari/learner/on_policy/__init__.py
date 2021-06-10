from .normal import NormalLearner


learner_collection = {
	'normal': NormalLearner
}

def create_learner(args):
	return learner_collection[args.learn](args)