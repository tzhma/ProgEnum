'''
for 268536 programs, single: 24m
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
from core_pte import gen_edit_distance_list, gen_perm_prog_array


## functions
def gen_min_edit_distance_to_enumDB(prog_id, df_enumP, df_enumDB):
	# load program
	program, progsize = df_enumP.loc[prog_id, ['program', 'progsize']].values

	# load DB_list with progsize
	DB_list = df_enumDB[df_enumDB['progsize']==progsize]['program'].values

	# permute program
	perm_prog_arr = gen_perm_prog_array(program)

	# find minimum edit distance to enumDB in DB_list
	d_list = gen_edit_distance_list(perm_prog_arr, DB_list)[0]
	d_min = min(d_list)
	return d_min


## load
df_task = pd.read_pickle('data/df_task')
df_enumP = pd.read_pickle('data/df_enumP_para14')
df_enumDB = pd.read_pickle('data/df_enumDB_para14')


## find min distance from program to enumDBs
if False:
	d_min_list = []
	for prog_id in df_enumP.index:
		d_min = gen_min_edit_distance_to_enumDB(prog_id, df_enumP, df_enumDB)
		d_min_list.append(d_min)
		if prog_id % max(len(df_enumP)//100,1)==0: print(prog_id, d_min)

	# df & override
	df_enumP['d2enumDB'] = d_min_list
	pd.to_pickle(df_enumP, 'data/df_enumP_para14')

	# plot
	plt.hist(d_min_list, bins=np.arange(10), rwidth=.9)
