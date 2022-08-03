'''
NOTE:
1) gen_flip_prog_list is to flip a+/a- so that boomerang plots are skew towards the same side
2) flip prog is no longer adopted after symmetrized SDP is used.
3) symmetrized SDP generates ao_hist that sums over a prog & its a-flipped version
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
from core_enum import fl
from core_bdp import run_bdp
from core_sdp import run_sdp
import multiprocess
import sparse

num_cpus = int(multiprocess.cpu_count()*1)
print('num_cpus = %s'%num_cpus)
if not os.path.exists('data'): os.makedirs('data')


## PARAMETERS
max_iter_sdp = 10


## LOAD
df_task = pickle.load(open('data/df_task','rb')) # 330 task parameters
para = df_task.loc[14, ['h','dp','dpm']].values # h,dp,dpm = para
df_gpn = pd.read_pickle('data/df_gpn')
df_enumP = pd.read_pickle('data/df_enumP')


# load GPN
tar_list = df_gpn['target'].values
sou_list = df_gpn['source'].values
id_uni_list = df_gpn['id_unique'].values
prog_list = df_enumP.loc[id_uni_list, 'program'].values


## FUNCTIONS: pre
def gen_flip_list(p_arr_list, prog_list):
	'''
	flip programs so that p(a+)>p(a-) is always true
	882 (20%) out of 4492 programs got flipped
	'''
	flip_list = []
	for i in range(len(tar_list)):
		pm = (p_arr_list[i]).sum(0)
		m0 = np.where(np.array(prog_list[i][0])==-1)
		m1 = np.where(np.array(prog_list[i][0])==1)
		pa0, pa1 = np.around(pm[m0].sum(),6), np.around(pm[m1].sum(),6)
		if pa0<=pa1: flip_list.append(0) # no flip
		else: flip_list.append(1) # flip
	return flip_list

def gen_flip_prog_list(flip_list, prog_list):
	flip_prog_lis = []
	for i in range(len(tar_list)):
		outmap, inmap = prog_list[i]
		if flip_list[i]==0: outmap1 = outmap
		else: outmap1 = tuple([-x for x in outmap])
		flip_prog_lis.append((outmap1, inmap))
	return flip_prog_lis


## FUNCTIONS: symmetrize ao_hist
def gen_Y_sym_idx(depth):
	y0 = {tuple(fl(x)):i for i,x in enumerate(product([(1, -1), (1, 1), (-1, -1), (-1, 1)], repeat=depth))}
	y1 = {k:y0[k] for k in sorted(y0)}
	return list(y1.values())


## MP: run_bdp (10m)
if False:
	def run_mp1(prog):
		eR, p_arr, t_iter = run_bdp(prog, para)
		return eR, p_arr

	eR_list, p_arr_list = [], []
	with multiprocess.Pool(num_cpus) as p:
		jobs = p.map(run_mp1, prog_list)

		for i, job in enumerate(jobs):
			eR_list.append(job[0])
			p_arr_list.append(job[1])

			if i % max(len(prog_list)//100, 1)==0:
				print(i, job[0])

	flip_list = gen_flip_list(p_arr_list, prog_list)
	flip_prog_list = gen_flip_prog_list(flip_list, prog_list)

	# pickle
	df_gpn['p_arr'] = p_arr_list
	df_gpn['flip_prog'] = flip_list
	df_gpn.to_pickle('data/df_gpn')
	pickle.dump(flip_prog_list, open('data/df_gpn_flip_prog_list', 'wb'))


## MP: run_sdp (15m for max_iter_sdp=10)
if False:
	def run_mp2(arg):
		p_arr, prog = arg
		Y_list = run_sdp(p_arr, prog, para, t_iter_max=max_iter_sdp)
		return Y_list

	Y_lists = []
	with multiprocess.Pool(num_cpus) as p:
		jobs = p.map(run_mp2, [(x,y) for x,y in zip(p_arr_list, flip_prog_list)])

		for i, job in enumerate(jobs):
			Y_lists.append(job)

			if i % max(len(flip_prog_list)//100, 1)==0: print(i, len(job[-1]))

	# pickle.dump(Y_lists, open('data/df_gpn_Y_lists_0', 'wb'))


## MP: symmetrize ao_hist (2m)
if False:
	# Y_lists = pickle.load(open('data/df_gpn_Y_lists_0', 'rb'))
	Y_symm_idx_list = [gen_Y_sym_idx(i) for i in range(1,11)]

	def run_mp3(Y_list_0):
		'''
		gen_Y_list_sym
		'''
		Y_list = [(x.todense() + x.todense()[Y_symm_idx_list[i]])/2 for i,x in enumerate(Y_list_0)]
		Y_list = [sparse.COO(x) for x in Y_list]
		return Y_list

	Y_lists_symm = []
	with multiprocess.Pool(num_cpus) as p:
		jobs = p.map(run_mp3, Y_lists)

		for i, job in enumerate(jobs):
			Y_lists_symm.append(job)

			if i % max(len(Y_lists)//100, 1)==0:
				print(i, len(job[-1]))

	# pickle
	pickle.dump(Y_lists_symm, open('data/df_gpn_Y_lists', 'wb'))
