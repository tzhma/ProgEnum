'''
NOTE: this is to evaluate simple DB rather than enumDB
cluster: 25m
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
import multiprocessing


## function
def fl(l):
	return [x for y in l for x in y]

def gen_eR_list(para_id):
	# load
	df = pd.read_pickle('data/df_task')
	para = df.loc[para_id, ['h','dp','dpm']].values
	DB_list = df.loc[para_id, 'DB_list']
	eR_B = df.loc[para_id, 'eR']

	# find eR_list
	eR_list, t_iter_list, performance_list = [], [], []
	for i, program in enumerate(DB_list):
		if i % max(len(DB_list)//5,1)==0: print('DB',i)
		program = program[:2]
		eR, p_array, t_iter = run_bdp(program, para)
		eR_list.append(eR)
		t_iter_list.append(t_iter)
		performance_list.append(eR/eR_B)
	return eR_list, t_iter_list, performance_list

def gen_DBsize_lists_for_990_995_999fullB(performance_lists):
	DBsize_lists = []
	for para_id, performance_list in enumerate(performance_lists):
		print('processing para', para_id)
		DB_list = pd.read_pickle('data/df_task').loc[para_id,'DB_list']
		DBsize_list = []
		for perf_thres in [.99, .995, .999]:
			idx_list = np.where(np.array(performance_list)>perf_thres)
			DBsizes = np.array([len(x[0]) for x in DB_list])
			DBsize = DBsizes[idx_list][:1]
			if len(DBsize)==1: DBsize_list.append(DBsize[0])
			else: DBsize_list.append(None)
		DBsize_lists.append(DBsize_list)
	return DBsize_lists


## MP
if False:
	num_cpus = int(multiprocessing.cpu_count()*1)
	print('num_cpus = %s'%num_cpus)
	para_id_list = range(330)

	eR_lists, t_iter_lists, performance_lists = [], [], []
	with multiprocessing.Pool(num_cpus) as p:
		jobs = p.map(gen_eR_list, para_id_list)

		for para_id, job in zip(para_id_list, jobs):
			print(para_id)
			eR_list, t_iter_list, performance_list = job
			eR_lists.append(eR_list)
			t_iter_lists.append(t_iter_list)
			performance_lists.append(performance_list)

	# df
	df = pd.read_pickle('data/df_task')
	df['eR_DB_list'] = eR_lists
	df['t_iter_DB_list'] = t_iter_lists
	df['performance_DB_list'] = performance_lists
	df['DBsize_for_99'] = gen_DBsize_lists_for_990_995_999fullB(performance_lists)
	pd.to_pickle(df, 'data/df_task')


# ##%% RUN: single
# plt.plot([y for x,y in zip(df.loc[14, 'DB_list'], df.loc[14, 'eR_DB_list']) if len(x[0])<=5])

# eR_lists, t_iter_lists, performance_lists = [], [], []
# for para_id in range(2):
# 	print(para_id)
# 	eR_list, t_iter_list, performance_list = gen_eR_list(para_id)
# 	eR_lists.append(eR_list)
# 	t_iter_lists.append(t_iter_list)
# 	performance_lists.append(performance_list)
