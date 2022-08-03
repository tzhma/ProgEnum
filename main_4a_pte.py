'''
mp: 1m20s for 5108
mp for 268533 programs: 5h, wt: 245h
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
import networkx as nx
import quantecon as qe
from collections import ChainMap
from core_bdp import run_bdp
from core_pte import find_prog_with_d1
import multiprocess

num_cpus = int(multiprocess.cpu_count()*1)
print('num_cpus = %s'%num_cpus)


## parameters
df_sort_type = 0		# 0:sorted, 1:half-randomized, 2:randomized (in eR)
num_top_P2s_kept = 2 	# 1, 2, 5
progsize_thres = 5 		# 4, 5
nth_found = 1

extension = ('5108','5105','268533')[1]
filename = 'df_pte_' + extension


## load
para = pickle.load(open('data/df_param', 'rb')).loc[14, ['h','dp','dpm']].values
eR_rand = .5 +.5*para[-1]

df = (
	pd.read_pickle('data/df_enumP_para14_sorted'),
	pd.read_pickle('data/df_enumP_para14_halfrand'),
	pd.read_pickle('data/df_enumP_para14_rand'),
)[df_sort_type]

prog_id_excluded = df[(df['progsize']==2)].sort_values('eR', ascending=False).index.values[num_top_P2s_kept:]
dff = df[(df['progsize']<=progsize_thres) & ([x not in prog_id_excluded for x in df.index])]


##%% MP: find n-th d=1 connection-program
def run(prog_id, nth_found=nth_found):
	prog_id_0s = []
	if dff.loc[prog_id, 'progsize']==2:
		row = find_prog_with_d1(prog_id, dff)
		return row
	for i in range(nth_found):
		if i==0:
			row = find_prog_with_d1(prog_id, dff)
			prog_id, prog_id_0, part, d_hist = row
			prog_id_0s.append(prog_id_0)
		else:
			row = find_prog_with_d1(prog_id, dff, first_prog_id_0=prog_id_0s[-1]+1)
			prog_id, prog_id_0, part, d_hist = row
			prog_id_0s.append(prog_id_0)
	return row

if False:
	prog_id_list = dff.index
	query = []
	with multiprocess.Pool(num_cpus) as p:
		jobs = p.map(run, prog_id_list)
		for prog_id, job in zip(prog_id_list, jobs):
			query.append(job)
			if prog_id % max(prog_id_list[-1]//100,1)==0: print(job)

	# df
	df_dat = pd.DataFrame(query, columns=['target', 'source', 'part', 'd_hist'])
	dff.loc[-1] = [((0,), [[0],[0]]), 0, 0, 0, -1, -1] # add empty entry for escaped programs
	df_dat['eR'] = dff.loc[df_dat['target'], 'eR'].values.astype('float')
	df_dat['eR_source'] = dff.loc[df_dat['source'], 'eR'].values.astype('float')
	df_dat['progsize'] = dff.loc[df_dat['target'], 'progsize'].values.astype('int')
	df_dat['d2enumDB'] = dff.loc[df_dat['target'], 'd2enumDB'].values.astype('int')
	df_dat['id_unique'] = dff.loc[df_dat['target'], 'id_unique'].values.astype('int')

	# append num_outedge to df
	num_outedge_arr = np.zeros(df_dat['target'].max()+1)
	for prog_id, prog_id_0 in df_dat[['target', 'source']].values:
		num_outedge_arr[prog_id_0] += 1
	df_dat['num_outedge'] = num_outedge_arr[df_dat['target']].astype('int')
	df_dat['id'] = df_dat['target'].values
	pd.to_pickle(df_dat, 'data/%s'%filename)

	# csv for gephi
	if not os.path.exists('gephi'): os.makedirs('gephi')
	df_dat.to_csv('gephi/%s.csv'%filename, index=False)
