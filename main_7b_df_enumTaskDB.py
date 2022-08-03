'''
3h on cluster
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
from core_enumDB import gen_DB_list_with_discretization_sweep
from core_enum import gen_m_groups_dict
from core_bdp import run_bdp
from core_bdpB import run_bdpB
import multiprocessing

num_cpus = int(multiprocessing.cpu_count()*1)
print('num_cpus = %s'%num_cpus)


## PARAMETERS: task parameter sweep
dp, dpm = .3, -.5
# h_list = np.around(np.linspace(.05,.40,36), 6) # coarse sweep
# h_list = np.around(np.linspace(.32, .37, 26), 6) # fine sweep
h_list = np.around(np.linspace(.05, .32, 28), 6)[:-1].tolist() + np.around(np.linspace(.32, .4, 41), 6).tolist()


## FUNCTIONS
def gen_df_enumTaskDB():
	'''
	45s
	'''
	## get m_groups_dict for merger ruleout
	prog_size_max = 30
	try: m_groups_dict = pickle.load(open('data/m_groups_dict_enumTaskDB','rb'))
	except FileNotFoundError:
		print('generating m_groups_dict (24m)')
		m_groups_dict = gen_m_groups_dict(prog_size_max=prog_size_max, for_enumDB=True) # 24m
		pickle.dump(m_groups_dict, open('data/m_groups_dict_enumTaskDB', 'wb'))

	## collect DBs from each task (upto progsize=30)
	DB_list = []
	for h in h_list:
		para = h, dp, dpm

		DB_list += gen_DB_list_with_discretization_sweep(para, m_groups_dict, DB_size_max=prog_size_max, use_merger_rule=True)
		print(h, len(DB_list))

	## filter out repeated
	DB_dict = {(tuple(x[0]), tuple(x[1].flatten())):x for x in DB_list}
	DB_list_r = list(DB_dict.values())
	DBsize_list_r = [len(x[0]) for x in DB_list_r]
	idx_sort = np.argsort(DBsize_list_r)
	DB_list_r = [DB_list_r[x] for x in idx_sort]
	DBsize_list_r = [DBsize_list_r[x] for x in idx_sort]

	# df
	df = pd.DataFrame([DB_list_r, DBsize_list_r]).T
	df.columns = ['program', 'progsize']
	print(len(df), 'unique DBs found')
	return df


## RUN: enum
if True:
	df = gen_df_enumTaskDB()
	df.to_pickle('data/df_enumTaskDB')


## MP: eval, 68 jobs
if False:
	df = pd.read_pickle('data/df_enumTaskDB')
	job_id = int(sys.argv[1])
	h = h_list[job_id-1]

	def run_eval(prog):
		'''
		20s for DBsize=2
		3m15s for DBsize=30
		'''
		para = h, dp, dpm
		eR, p_arr, t_iter = run_bdp(prog, para, dim_u=180)
		return eR, p_arr, t_iter
	#
	DB_list = df['program'].values
	eR_list, t_iter_list, p_arr_list = [], [], []
	with multiprocessing.Pool(num_cpus) as p:
		jobs = p.map(run_eval, DB_list)

		for k, job in enumerate(jobs):
			eR_list.append(job[0])
			p_arr_list.append(job[1])
			t_iter_list.append(job[2])
			print(k, np.mean(job[0]))

	# df
	df['eR'] = eR_list
	df['t_iter'] = t_iter_list
	df['p_arr'] = p_arr_list

	# pickle
	pickle.dump(df, open('data/df_enumTaskDB_%s'%job_id, 'wb'))


## ASSEMBLE 68 jobs
if False:
	progsize_list = pd.read_pickle('data/df_enumTaskDB')['progsize'].values
	eR_lists = []
	for job_id in range(1,69):
		df = pd.read_pickle('data/df_enumTaskDB_%s'%job_id)
		eR_lists.append(df['eR'].values)
	eR_arr = np.array(eR_lists).T

	# pickle
	pickle.dump((progsize_list, h_list, eR_arr), open('data/df_enumTaskDB_data', 'wb'))
