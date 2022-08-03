'''
cluster: 13m
'''
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
import warnings
warnings.filterwarnings('ignore')
import time
from itertools import permutations, combinations, product, islice
import pandas as pd
from core_bdp import run_bdp
import multiprocess

num_cpus = int(multiprocess.cpu_count()*1)
print('num_cpus = %s'%num_cpus)


## LOAD
df_task = pickle.load(open('data/df_task','rb')) # 330 task parameters
df_enumDB = pickle.load(open('data/df_enumDB_para14','rb'))
program_list = df_enumDB['program'].values

## world parameters
para_id = 14
para = df_task.loc[para_id, ['h','dp','dpm']].values # h,dp,dpm = para
eR_B = df_task.loc[para_id, 'eR_B']


## MP
def run(prog_id):
	program = program_list[prog_id]
	eR, p_array, t_iter = run_bdp(program, para)
	return eR, t_iter

if False:
	prog_id_list = range(len(program_list))
	eR_list, t_iter_list = [], []
	with multiprocess.Pool(num_cpus) as p:
		jobs = p.map(run, prog_id_list)

		for prog_id, job in zip(prog_id_list, jobs):
			eR_list.append(job[0])
			t_iter_list.append(job[1])

			if prog_id % max(len(program_list)//100, 1)==0:
				print('program', prog_id, job)

	# df
	df = pd.read_pickle('data/df_enumDB_para14')
	df['eR'] = eR_list
	df['t_iter'] = t_iter_list
	pd.to_pickle(df, 'data/df_enumDB_para14')


##%% PLOT
print(len(df[(df['progsize']<=5) & (df['eR']>=.277911)]))
plt.hist(df[df['progsize']<=5]['eR'], bins=100)
