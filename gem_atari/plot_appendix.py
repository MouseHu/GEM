import os
import copy
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
from utils.os_utils import get_arg_parser, dir_ls
from plot import task_y_axis, plot_colors, plot_colors_alg, final_id, get_data, plot_legend

task_rews_y_axis = {
	'Alien': (-100.0, 2100.0),
	'Assault': (-100.0, 2000.0),
	'Asterix': (-200.0, 6200.0),
	'Atlantis': (-10000.0, 1100000.0),
	'BankHeist': (-50.0, 1300.0),
	'BattleZone': (-1500.0, 31500.0),
	'BeamRider': (-200.0, 7700.0),
	'Bowling': (-3.0, 53.0),
	'Enduro': (-100.0, 1800.0),
	'Frostbite': (-300.0, 6300.0),
	'Jamesbond': (-20.0, 550.0),
	'Krull': (-300.0, 9000.0),
	'MsPacman': (-150.0, 3500.0),
	'Qbert': (-300.0, 9000.0),
	'Riverraid': (-500.0, 14000.0),
	'RoadRunner': (-2000.0, 57000.0),
	'StarGunner': (-2000.0, 32000.0),
	'TimePilot': (-200.0, 6500.0),
	'WizardOfWor': (-50.0, 900.0),
	'Zaxxon': (-500.0, 11000.0),
}

if __name__=='__main__':
	parser = get_arg_parser()
	parser.add_argument('--smooth', help='number of checkpoints for smoothing', type=np.int32, default=10)
	parser.add_argument('--color', help='color theme id', type=np.int32, default=3)
	args = parser.parse_args()

	plot_colors_alg = plot_colors_alg[args.color]

	matplotlib.use('Agg')
	plt.figure(figsize=(15.3,25.05),frameon=False)

	CI_level = 0.6
	dir_List = [
		'Rews/alien-rews',
		#'Rews/asterix-rews',
		'Rews/atlantis-rews',
		'Rews/bankheist-rews',
		'Rews/battlezone-rews',
		'Rews/beamrider-rews',
		'Rews/bowling-rews',
		'Rews/enduro-rews',
		'Rews/frostbite-rews',
		'Rews/jamesbond-rews',
		'Rews/krull-rews',
		'Rews/mspacman-rews',
		'Rews/qbert-rews',
		'Rews/riverraid-rews',
		'Rews/roadrunner-rews',
		'Rews/stargunner-rews',
		'Rews/timepilot-rews',
		'Rews/wizardofwor-rews',
		'Rews/zaxxon-rews',
	]
	lines = []

	for fig_id, dir_id in enumerate(dir_List):
		length = 10000000
		title, final_flag = None, False
		length //= 50000
		data, data_path = get_data(dir_id, final_flag, args.smooth)

		for task in data.keys():
			plt.style.use('seaborn-whitegrid')
			plt.rc('font', family='Times New Roman')
			plt.subplot(6, 3, fig_id+1)
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
				if alg in plot_colors_alg.keys():
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
	plt.figlegend(handles=lines, loc='upper center', prop={'size':22.0}, shadow=True, ncol=3)
	plt.tight_layout(rect=(0,0,1,1.0-0.8/25.05))
	plt.savefig('plot_data/paper/appendix.pdf', bbox_inches='tight')
	plt.close()
