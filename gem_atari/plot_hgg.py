import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from plot import final_id, get_data
from utils.os_utils import get_arg_parser, dir_ls

experiments_set = {
	'1': ['1_3','1_4'],
	'2': ['2_2', '2_3', '2_4'],
	'3': ['3_1', '3_3', '3_4'],
	'23': ['2_2', '2_3', '2_4', '3_2', '3_3', '3_4'],
	'4': ['4_1'],
	'5': ['5_1', '5_2', '5_3'],
	'6_1': ['6_1'],
	'6_2': ['6_2'],
	'7': ['7_1', '7_2', '7_3', '7_4', '7_5', '7_6', '7_7', '7_8'],
	'7_slide': ['7_2', '7_3', '7_6', '7_7'],
	'8': ['8_1'],
	'9': ['9_1','9_3','9_5','9_7'],
	'9_full': ['9_1','9_3','9_5','9_2','9_4','9_6'],
	'10': ['10_1','10_2','10_3','10_4'],
	'11': ['11_1'],
	'12': ['12_1'],
	'13': ['13_1']
}

main = np.array([1.0,0.0,0.0])
main_2 = np.array([1.0,0.5,0.0])
match = np.array([0.7,0.0,0.4])
EBP =  np.array([0.0,0.0,1.0])
HER = np.array([0.0,0.7,0.7])

main_experiment_color = {
	'HER+EBP+HGG': main,
	# 'HER+match': match,
	'HER+HGG': main_2,
	'HER+EBP': np.array([0.0,0.0,1.0]),
	'HER': np.array([0.0,0.7,0.7]),
	'test': np.array([0,0,0])
}

ablation_experiment_color = {
	"L=0.5": np.array([0.3,0.4,0.5]),
	"L=1.0": np.array([0.2,0.7,0.2]),
	"L=2.0": np.array([0.0,0.5,0.7]),
	"L=5.0": main_2,
	"L=10.0": np.array([0.7,0.2,0.7]),
	'c=0.5': np.array([0.3,0.4,0.5]),
	'c=1.5': np.array([0.2,0.7,0.2]),
	'c=3.0': main_2,
	'c=5.0': np.array([0.0,0.5,0.7]),
	'K=1': np.array([0.3,0.4,0.5]),
	'K=25': np.array([0.2,0.7,0.2]),
	'K=50': np.array([0.0,0.5,0.7]),
	'K=100': main_2,
	'K=200': np.array([0.7,0.2,0.7]),
	'test': np.array([0,0,0])
}

oracle_experiment_color = {
	# 'HER+oracle': np.array([0.5,0.7,0.0]),
	# 'HER+greedy': np.array([0.7,0.2,0.7]),
	'HER+GOID(alpha=0.1)': np.array([0.5,0.7,0.0]),
	'HER+GOID(alpha=0.2)': np.array([0.5,0.0,0.7]),
	'HGG': main_2,
	'HER': np.array([0.0,0.5,0.7]),
}

final_colors = {
	'1': {
		'0.0': np.array([0.5,0.7,0.0]),
		'0.5': np.array([0.7,0.2,0.7]),
		'1.0': np.array([0.0,0.5,0.7]),
		'1.5': np.array([0.5,0.7,0.2]),
		'2.0': np.array([0.7,0.5,0.2]),
		'3.0': main,
		'5.0': np.array([0.5,0.2,0.2]),
		'HER(baseline)': HER,
	},
	'2': main_experiment_color,
	'3': main_experiment_color,
	'23': main_experiment_color,
	'4': {
		'HER+EBP+greedy': main,
		'HER+EBP+match': match
	},
	'5': {
		'25': np.array([0.5,0.7,0.0]),
		'50': np.array([0.7,0.2,0.7]),
		'100': np.array([0.0,0.5,0.7]),
		'200': np.array([0.5,0.7,0.2]),
	},
	'6_1': oracle_experiment_color,
	'6_2': oracle_experiment_color,
	'7': main_experiment_color,
	'7_slide': main_experiment_color,
	'8': {
		'0.0': np.array([0.5,0.7,0.0]),
		'0.5': np.array([0.7,0.2,0.7]),
		'1.0': np.array([0.0,0.5,0.7]),
		'1.5': np.array([0.5,0.7,0.2]),
		'2.0': np.array([0.7,0.5,0.2]),
		'3.0': main,
		'5.0': np.array([0.5,0.2,0.2]),
		'HER(baseline)': HER,
	},
	'9': ablation_experiment_color,
	'9_full': ablation_experiment_color,
	'10': main_experiment_color,
	'11': {
		'HGG': main_2,
		'subset': np.array([0.0,0.7,0.7]),
	},
	'12': {
		'HGG+grid distance': main,
		'HGG+l2 distance': main_2,
		'HER': np.array([0.0,0.5,0.7]),
		#'test': np.array([0,0,0])
	},
	'13': {
		'HGG': main_2,
		'HER': np.array([0.0,0.7,0.7]),
	}
}

size_dict = {
	(2,1): (10.0, 4.5),
	(2,2): (10.0, 8.25),
	(3,1): (15.3, 4.5),
	(3,2): (15.3, 8.25),
	(1,1): (5.3, 4.3),
	(4,2): (20.3, 8.25),
	(4,1): (20.3, 4.3)
}

def convert_name(pre_name):
	if pre_name == 'HER+HGG': return 'HGG'
	if pre_name == 'HER+EBP+HGG': return 'HGG+EBP'
	if pre_name == 'HGG+l2 distance': return 'HGG+'+r'$\ell_2$'+' distance'
	for i in range(len(pre_name)):
		if pre_name[i:i+5]=='alpha':
			return pre_name[:i]+'\u03B1'+pre_name[i+5:]
	return pre_name

def get_alg_list(exp_id):
	'''
	if exp_id[0]=='9':
		return [
			"L=5.0", "L=0.5", "L=1.0", "L=2.0", "L=10.0",
			'c=3.0', 'c=0.5', 'c=1.5', 'c=5.0',
			'K=100', 'K=1', 'K=25', 'K=50', 'K=200',
			'test'
		]
	else:
		return list(final_colors[args.id].keys())
	'''
	return list(final_colors[args.id].keys())

if __name__=='__main__':
	parser = get_arg_parser()
	parser.add_argument('--id', help='plot experiment id', type=str)
	parser.add_argument('--CI', help='confidence interval', type=np.float32, default=0.6)
	args = parser.parse_args()

	matplotlib.use('Agg')
	fig_cnt = len(experiments_set[args.id])
	if fig_cnt==8: fig_w, fig_h = 4, 2
	elif fig_cnt==4:
		if args.id=='7_slide': fig_w, fig_h = 2, 2
		else: fig_w, fig_h = 4, 1
	else: fig_w, fig_h = (fig_cnt-1)%3+1, (fig_cnt-1)//3+1
	size_w, size_h = size_dict[(fig_w,fig_h)]
	plt.figure(figsize=(size_w,size_h),frameon=False)

	CI_level = args.CI
	legend_id, legend_cnt, lines = 0, 0, []
	for fig_id, experiment in enumerate(experiments_set[args.id]):
		alg_cnt = 0
		data, _ = get_data(final_id[experiment][0])
		if len(list(data.keys()))==0: continue
		task = list(data.keys())[0]
		for alg in list(final_colors[args.id].keys()):
			if alg in data[task].keys():
				alg_cnt += 1
		if alg_cnt>legend_cnt:
			legend_cnt = alg_cnt
			legend_id = fig_id

	for fig_id, experiment in enumerate(experiments_set[args.id]):
		if args.id[0]=='9': lines = []
		data, _ = get_data(final_id[experiment][0])
		plt.subplot(fig_h, fig_w, fig_id+1)
		if args.id in ['6_2']:
			plt.title(final_id[experiment][1],size=16.0)
		else:
			plt.title(final_id[experiment][1],size=18.0)
		plt.ylim((-0.05, 1.05))
		plt.xticks(fontsize=10.0)
		plt.yticks(fontsize=12.0)
		plt.xlabel('episodes', size=20.0)
		if fig_id%fig_w==0: plt.ylabel('median success rate', size=20.0)
		if len(list(data.keys()))==0: continue
		assert len(data.keys())==1
		task = list(data.keys())[0]
		for alg in get_alg_list(args.id)[::-1]:
			if not(alg in data[task].keys()):
				continue
			cnt = len(data[task][alg])
			Len = min([len(record) for record in data[task][alg]])
			X, Y, Y_l, Y_r = [], [], [], []
			for i in range(Len):
				if Len>12000//50:
					if data[task][alg][0][i][0]%100==0:
						X.append(data[task][alg][0][i][0])
						now_y = []
						for record in data[task][alg]:
							assert record[i][0]==X[-1]
							now_y.append((record[i][1]+record[max(i-1,0)][1])/2.0)
						now_y = np.array(now_y)
						now_y.sort()
						Y.append(np.median(now_y))
						d = int(cnt*(1-CI_level)/2+1e-3)
						assert abs(d-cnt*(1.0-CI_level)/2)<1e-2
						Y_l.append(now_y[d])
						Y_r.append(now_y[cnt-1-d])
				else:
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

			alg_color = final_colors[args.id][alg]
			length = final_id[experiment][2]//(100 if Len>12000//50 else 50)
			'''
			if (experiment=='3_2' or experiment=='7_3') and alg=='HER+HGG':
				line, = plt.plot(X[:length], np.array(Y[:length])-0.01, color=alg_color, label=convert_name(alg))
			else:
			'''
			line, = plt.plot(X[:length], Y[:length], color=alg_color, label=convert_name(alg))
			if fig_id==legend_id or args.id[0]=='9': lines.append(line)
			plt.fill_between(X[:length], Y_l[:length], Y_r[:length], color=alg_color, alpha=0.1)

		if args.id[0]=='9':
			plt.legend(handles=lines[::-1], loc='best', prop={'size':13.0}, shadow=True, ncol=1)

	if args.id[0]!='9':
		if fig_w*fig_h==1:
			plt.legend(handles=lines[::-1], loc='upper left', prop={'size':14.0}, framealpha=0.5, shadow=False, ncol=1)
			plt.tight_layout(rect=(0,0,1,1))
		else:
			plt.figlegend(handles=lines[::-1], loc='upper center', prop={'size':15.0}, shadow=True, ncol=100)
			plt.tight_layout(rect=(0,0,1,1.0-0.5/size_h))
	else:
		plt.tight_layout(rect=(0,0,1,1))
	plt.savefig('plot_data/'+args.id+'.png')

	plt.close()
