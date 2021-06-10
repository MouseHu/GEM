import json
import numpy as np
import copy
from utils.os_utils import make_dir

for env_name in ['Alien', 'BankHeist', 'BattleZone', 'Frostbite']:
    for alg_name in ['IQN', 'RAINBOW', 'C51', 'DQN']:

        with open("plot_data/json/"+env_name+".json",'r') as load_f:
            load_dict = json.load(load_f)
            print(type(load_dict[0]))

        cnt = 0
        for item in load_dict:
            if 'Agent' in item.keys():
                if item['Agent'] == alg_name:
                    if item['Iteration'] == 0:
                        record = {env_name: []}
                    record[env_name].append(((item['Iteration']+1)*250000, item['Value']))
                    if item['Iteration'] == 100:
                        make_dir('plot_data/json/'+env_name, False)
                        make_dir('plot_data/json/'+env_name+'/'+alg_name, False)
                        make_dir('plot_data/json/'+env_name+'/'+alg_name+'/'+str(cnt), False)
                        np.savez('plot_data/json/'+env_name+'/'+alg_name+'/'+str(cnt)+'/'+env_name+'.npz', info=copy.deepcopy(record[env_name]))
                        cnt += 1
