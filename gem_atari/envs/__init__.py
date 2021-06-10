import gym
import envs.atari as atari_env

envs_collection = {
	# Atari envs
	'Alien': 'atari',
	'Assault': 'atari',
	'Asterix': 'atari',
	'Atlantis': 'atari',
	'BankHeist': 'atari',
	'BattleZone': 'atari',
	'BeamRider': 'atari',
	'Bowling': 'atari',
	'Defender': 'atari',
	'DemonAttack': 'atari',
	'Enduro': 'atari',
	'Frostbite': 'atari',
	'Jamesbond': 'atari',
	'JourneyEscape': 'atari',
	'Krull': 'atari',
	'MsPacman': 'atari',
	'Pong': 'atari',
	'Qbert': 'atari',
	'Phoenix': 'atari',
	'Riverraid': 'atari',
	'RoadRunner': 'atari',
	'Solaris': 'atari',
	'StarGunner': 'atari',
	'TimePilot': 'atari',
	'WizardOfWor': 'atari',
	'Zaxxon': 'atari',
}

make_env_collection = {
	'atari': atari_env
}

def make_env(args):
	return make_env_collection[envs_collection[args.env]].make_env(args)
