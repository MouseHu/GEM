from .vanilla import VanillaEnv

def make_env(args):
	return VanillaEnv(args)