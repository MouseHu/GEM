import os
import copy
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
from utils.os_utils import get_arg_parser, dir_ls
from plot import task_rews_y_axis, task_y_axis, plot_colors, final_id, get_data

plot_colors_alg = [
	{
		'LBCDDQN': np.array([0.9,0.17,0.31]),
		'LBDDQN': np.array([1.0,0.5,0.0]),
		'CDDQN': np.array([0.56, 0.0, 1.0]),
		'DDQN': np.array([0.19,0.55,0.91]),
		'Maxmin DQN': np.array([0.56,0.74,0.56]),
		'DQN': np.array([0.66, 0.66, 0.66]),
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
		'n=7 + LB': np.array([0.9,0.17,0.31]),
		'n=5 + LB': np.array([0.9,0.17,0.31]),
		'n=3 + LB': np.array([0.9,0.17,0.31]),
		'n=1 + LB': np.array([1.0,0.5,0.0]),
		'n=7': np.array([0.66, 0.66, 0.66]),
		'n=5': np.array([0.56,0.74,0.56]),
		'n=3': np.array([0.56, 0.0, 1.0]),
		'n=1': np.array([0.19,0.55,0.91]),
	},
]


plot_legend = {
	'n=7 + LB': 'n*-step + LB',
	'n=5 + LB': 'n*-step + LB',
	'n=3 + LB': 'n*-step + LB',
	'n=1 + LB': '1-step + LB',
	'n=7': '7-step',
	'n=5': '5-step',
	'n=3': '3-step',
	'n=1': '1-step',
	'LBCDDQN': 'Clipped DDQN + LB (ours)',
	'LBDDQN': 'DDQN + LB (ours)',
	'CDDQN': 'Clipped DDQN',
	'DDQN': 'DDQN',
	'Maxmin DQN': 'Maxmin DQN',
	'DQN': 'DQN',
}

if __name__=='__main__':
	parser = get_arg_parser()
	parser.add_argument('--smooth', help='number of checkpoints for smoothing', type=np.int32, default=10)
	parser.add_argument('--color', help='color theme id', type=np.int32, default=4)
	args = parser.parse_args()

	plot_colors_alg = plot_colors_alg[args.color]

	matplotlib.use('Agg')
	plt.figure(figsize=(17.5, 4.25),frameon=False)

	CI_level = 0.6
	dir_List = [
		('paper/nstep/alien-nstep-rews', 5000000),
		('paper/nstep/stargunner-nstep-rews', 20000000),
		('paper/nstep/zaxxon-nstep-rews', 15000000),
	]
	lines = []

	for fig_id, dir_entry in enumerate(dir_List):
		dir_id, length = dir_entry
		title, final_flag = None, False
		length //= 50000
		data, data_path = get_data(dir_id, final_flag, args.smooth)

		for task in data.keys():
			plt.style.use('seaborn-whitegrid')
			plt.rc('font', family='Times New Roman')
			plt.subplot(1, 3, fig_id+1)
			# matplotlib.rcParams['text.usetex'] = True
			ax = plt.gca()
			ax.spines['right'].set_visible(False)
			ax.spines['top'].set_visible(False)
			ax.spines['left'].set_color('black')
			ax.spines['bottom'].set_color('black')
			if not (title is None):
				plt.title(title, size=24.0)
			else:
				plt.title(task, size=24.0)
			if dir_id[-5:]=='-rews':
				plt.ylim(task_rews_y_axis[task])
			else:
				plt.ylim(task_y_axis[task])
			plt.tick_params('x', labelsize=18.0)
			plt.tick_params('y', labelsize=14.0)
			plt.xlabel('Timesteps', size=20.0)
			ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
			ax.yaxis.set_major_locator(ticker.MaxNLocator(6))
			if fig_id%3==0:
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
				line, = plt.plot(X[:length], Y[:length], color=alg_color, label=plot_legend[alg], zorder=6-id)
				plt.fill_between(X[:length], Y_l[:length], Y_r[:length], color=alg_color, alpha=0.1)

				if fig_id==0:
					lines.append(line)

			#plt.legend(loc='best', prop={'size':22.0}, frameon=True, framealpha=1.0, facecolor='white', ncol=1)

	plt.figlegend(handles=lines, loc='center right', prop={'size': 18.0}, frameon=True, ncol=1)
	plt.tight_layout(rect=(0, 0, 15.3 / 17.5, 1))
	plt.savefig('plot_data/paper/nstep/nstep.pdf', bbox_inches='tight')
	plt.close()
