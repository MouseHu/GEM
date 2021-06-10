import os
import copy
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
from utils.os_utils import get_arg_parser, dir_ls

task_rews_y_axis = {
	'Alien': (-100.0, 2100.0),
	'Assault': (-100.0, 2000.0),
	'Asterix': (-100.0, 6000.0),
	'Atlantis': (-100.0, 1200000.0),
	'BankHeist': (-50.0, 1300.0),
	'BattleZone': (-1500.0, 31500.0),
	'BeamRider': (-100.0, 10000.0),
	'Bowling': (-2.0, 50.0),
	'Enduro': (-100.0, 2000.0),
	'Frostbite': (-300.0, 7800.0),
	'Jamesbond': (-100.0, 600.0),
	'Krull': (-100.0, 10000.0),
	'MsPacman': (-150.0, 4500.0),
	'Qbert': (-100.0, 10000.0),
	'Riverraid': (-100.0, 15000.0),
	'RoadRunner': (-2000.0, 55000.0),
	'StarGunner': (-2000.0, 52000.0),
	'TimePilot': (-100.0, 7200.0),
	'WizardOfWor': (-50.0, 900.0),
	'Zaxxon': (-500.0, 13000.0),
}

task_y_axis = {
	'Alien': (-2.0, 20.0),
	'Atlantis': (-2.0, 8.0),
	'StarGunner': (-2.0, 10.0),
	'TimePilot': (-0.5, 3.0),
}

plot_colors = [
	np.array([1.0,0.0,0.0]),
	np.array([1.0,0.5,0.0]),
	np.array([0.0,0.0,1.0]),
	np.array([0.5,0.0,1.0]),
	np.array([1.0,0.0,1.0]),
	np.array([0.0,1.0,0.0]),
	np.array([1.0,1.0,0.0]),
	np.array([0.0,1.0,1.0]),
	np.array([0.5,0.5,0.5]),
	np.array([1.0,0.0,0.5]),
	np.array([0.5,1.0,0.0]),
	np.array([0.0,1.0,0.5]),
	np.array([0.0,0.5,1.0]),
]

plot_colors_alg = [
	{
		'LBCDDQN': np.array([0.9,0.17,0.31]),
		'LBDDQN': np.array([1.0,0.5,0.0]),
		'CDDQN': np.array([0.56,0.0,1.0]),
		'DDQN': np.array([0.19,0.55,0.91]),
		'Dueling DDQN': np.array([0.38,0.25,0.32]),
		'Averaged DQN': np.array([0.8,0.36,0.27]),
		'Maxmin DQN': np.array([0.56,0.74,0.56]),
		'LBDQN': np.array([0.47,0.41,0.47]),
		'DQN': np.array([0.66,0.66,0.66]),
	},
	{
		'DDQN': np.array([1.0,0.49,0.0]),
		'DQN': np.array([0.19,0.55,0.91]),
	},
	{
		'LBDDQN': np.array([1.0,0.5,0.0]),
		'DDQN': np.array([0.19,0.55,0.91]),
		'DDQN 3-step': np.array([0.56, 0.0, 1.0]),
		'DDQN 5-step': np.array([0.56,0.74,0.56]),
		'DDQN 7-step': np.array([0.66, 0.66, 0.66]),
	},
	{
		'LBDDQN': np.array([1.0,0.5,0.0]),
		'DDQN': np.array([0.19,0.55,0.91]),
	},
	{
		'DQN': np.array([0.66,0.66,0.66]),
		'Dueling DDQN': np.array([0.38,0.25,0.32]),
		'rank=3': np.array([0.56,0.0,1.0]),
		'rank=5': np.array([0.19,0.55,0.91]),
	}
]

final_id = {
	'1_1': ('final/1_1_vanilla', 'FetchPickAndPlace(standard)', 8000),
	'1_2': ('final/1_2_interval', 'FetchPickAndPlace(segment)', 6000),
	'1_3': ('final/1_3_vanilla', 'FetchPush(standard)', 6000),
	'1_4': ('final/1_4_interval', 'FetchPush(segment)', 6000),
}

plot_legend = {
	'LBCDDQN': 'Clipped DDQN + LB (ours)',
	'LBDDQN': 'DDQN + LB (ours)',
	'CDDQN': 'Clipped DDQN',
	'DDQN': 'DDQN',
	'Dueling DDQN': 'Dueling DDQN',
	'Averaged DQN': 'Averaged DQN',
	'Maxmin DQN': 'Maxmin DQN',
	'LBDQN': 'DQN + LB',
	'DQN': 'DQN',
}

def get_data(curve_id, final_flag=True, smooth=10):
	data = {}
	data_path = 'plot_data/'+curve_id
	for alg in dir_ls(data_path):
		alg_path = data_path+'/'+alg
		if os.path.isdir(alg_path):
			alg_type = alg if final_flag else alg[2:]
			for run in dir_ls(alg_path):
				run_path = alg_path+'/'+run
				for record in dir_ls(run_path):
					record_path = run_path+'/'+record
					info_load = np.load(record_path)['info'][()]
					info, moving_ave = [], []
					for test_info_load in info_load:
						test_info = copy.deepcopy(test_info_load)
						moving_ave.append(test_info[1])
						if(len(moving_ave)>smooth):
							moving_ave.pop(0)
						test_info[1] = np.mean(moving_ave)
						info.append(test_info)
					record_type = record[:-4]
					if not(record_type in data.keys()):
						data[record_type] = {}
					if not(alg_type in data[record_type].keys()):
						data[record_type][alg_type] = []
					data[record_type][alg_type].append(info)
	return data, data_path

if __name__=='__main__':
	parser = get_arg_parser()
	parser.add_argument('--id', help='plot data id', type=str)
	parser.add_argument('--len', help='length of ploted curve', type=np.int32, default=10000000)
	parser.add_argument('--smooth', help='number of checkpoints for smoothing', type=np.int32, default=10)
	parser.add_argument('--color', help='color theme id', type=np.int32, default=0)
	parser.add_argument('--framealpha', help='legend frame alpha', type=np.float32, default=1.0)
	args = parser.parse_args()
	assert args.id!=None

	plot_colors_alg = plot_colors_alg[args.color]

	title, final_flag = None, False
	if args.id in final_id.keys():
		id, title, length = final_id[args.id]
		length //= 50000
		args.id = id
		final_flag = True
	else:
		length = args.len
		length //= 50000

	data, data_path = get_data(args.id, final_flag, args.smooth)

	matplotlib.use('Agg')
	for CI_level in [0.6,1.0]:
		for task in data.keys():
			if args.color!=1:
				plt.style.use('seaborn-whitegrid')
			plt.rc('font', family='Times New Roman')
			# matplotlib.rcParams['text.usetex'] = True
			plt.clf()
			ax = plt.gca()
			ax.spines['right'].set_visible(False)
			ax.spines['top'].set_visible(False)
			ax.spines['left'].set_color('black')
			ax.spines['bottom'].set_color('black')
			if not (title is None):
				plt.title(title, size=24.0)
			else:
				plt.title(task, size=24.0)
			if args.id[-5:]=='-rews':
				plt.ylim(task_rews_y_axis[task])
			else:
				plt.ylim(task_y_axis[task])
			plt.tick_params('x', labelsize=18.0)
			plt.tick_params('y', labelsize=14.0)
			plt.xlabel('Timesteps', size=22.0)
			plt.ylabel('Average Episode Return', size=22.0)

			def formatnum(x, pos):
				return str(int(x/1e6+0.001)) + 'M'  # '$%.1f$M' % (x / 1e6)
			formatter = FuncFormatter(formatnum)
			ax.xaxis.set_major_formatter(formatter)

			for id, alg in enumerate(data[task].keys()):
				cnt = len(data[task][alg])
				Len = min([len(record) for record in data[task][alg]])
				X, Y, Y_l, Y_r = [], [], [], []
				for i in range(Len):
					X.append(data[task][alg][0][i][0])
					now_y = []
					for record in data[task][alg]:
						assert record[i][0]==X[i]
						now_y.append(record[i][1])
					now_y = np.array(now_y)
					now_y.sort()
					Y.append(np.median(now_y))
					d = int(cnt*(1-CI_level)/2+1e-3)
					assert abs(d-cnt*(1.0-CI_level)/2)<1e-2
					Y_l.append(now_y[d])
					Y_r.append(now_y[cnt-1-d])

				# alg_color = plot_colors[id]
				alg_color = plot_colors_alg[alg]
				plt.plot(X[:length], Y[:length], color=alg_color, label=alg, zorder=len(list(plot_colors_alg.keys()))-id)
				plt.fill_between(X[:length], Y_l[:length], Y_r[:length], color=alg_color, alpha=0.1)

			plt.legend(loc='best', prop={'size':22.0}, frameon=True, framealpha=args.framealpha, facecolor='white', ncol=1)
			plt.savefig(data_path+'/'+task+'_'+str(CI_level)+'.pdf', bbox_inches='tight')

	plt.close()
