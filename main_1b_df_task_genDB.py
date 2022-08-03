'''
NOTE: This is simple DB rather than enumDB
mp: 3m for DB_size_max=8
mp: 11m for DB_size_max=20
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
import multiprocess

num_cpus = int(multiprocess.cpu_count()*1)
print('num_cpus = %s'%num_cpus)


## LOAD
df = pd.read_pickle('data/df_task')
m_groups_dict = gen_m_groups_dict(prog_size_max=20, for_enumDB=True)
# m_groups_dict = pickle.load(open('data/m_groups_dict_enumDB','rb'))


## MP
def run(para_id):
	'''
	discretize DB with a swept range of belief values
	'''
	para = df.loc[para_id, ['h','dp','dpm']]
	DB_list = gen_DB_list_with_discretization_sweep(para, m_groups_dict, DB_size_max=20)
	return DB_list

if False:
	para_id_list = df.index
	DB_lists = []
	with multiprocess.Pool(num_cpus) as p:
		jobs = p.map(run, para_id_list)

		for para_id, job in zip(para_id_list, jobs):
			DB_lists.append(job)
			print(para_id, len(job))

	# pickle
	df['DB_list'] = DB_lists
	pd.to_pickle(df, 'data/df_task')


##%% PLOT
for para_id in range(15):
	counts = [len(x[0]) for x in df.loc[para_id,'DB_list']]
	plt.figure()
	plt.title('%s, %s'%(para_id, len(counts)))
	plt.hist(counts, bins=np.arange(2,22)-.5, rwidth=.8)
