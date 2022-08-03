'''
run this fully (268,536 programs @ param14) on cluster
mp on cluster: 1.8hrs, wt:91hrs
'''
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import os
import sys
from itertools import permutations, combinations, product
import pandas as pd
from core_bdp import run_bdp
import multiprocessing

num_cpus = int(multiprocessing.cpu_count()*1)
print('num_cpus = %s'%num_cpus)


## LOAD
df_task = pickle.load(open('data/df_task','rb')) # 330 task parameters
df_enumP = pickle.load(open('data/df_enumP','rb'))
program_list = df_enumP['program'].values

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
	with multiprocessing.Pool(num_cpus) as p:
		jobs = p.map(run, prog_id_list)

		for prog_id, job in zip(prog_id_list, jobs):
			eR_list.append(job[0])
			t_iter_list.append(job[1])

			if prog_id % max(len(program_list)//100, 1)==0:
				print('program', prog_id, job)

	# df
	df = pd.read_pickle('data/df_enumP')
	df['eR'] = eR_list
	df['t_iter'] = t_iter_list
	pd.to_pickle(df, 'data/df_enumP_para%s'%para_id)
