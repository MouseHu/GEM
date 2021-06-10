import learner.on_policy as on_policy
import learner.off_policy as off_policy

learner_collection = {**on_policy.learner_collection, **off_policy.learner_collection}

def create_learner(args):
	if args.on_policy:
		return on_policy.create_learner(args)
	elif args.off_policy:
		return off_policy.create_learner(args)
	else:
		assert int(args.on_policy)+int(args.off_policy)==1