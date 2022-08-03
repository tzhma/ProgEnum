'''
...
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
from core_sdp import run_sdp
from core_enum import fl
import sparse
import multiprocessing


num_cpus = int(multiprocessing.cpu_count()*1)
print('num_cpus = %s'%num_cpus)
if not os.path.exists('data/df_enumTaskP'): os.makedirs('data/df_enumTaskP')


## PARAMETERS
dp, dpm = .3, -.5
h_list = np.around(np.linspace(.05, .32, 28), 6)[:-1].tolist() + np.around(np.linspace(.32, .4, 41), 6).tolist()

job_id = int(sys.argv[1]) # 1~68
h = h_list[job_id-1]
para = h, dp, dpm


## LOAD
try:
	df = pd.read_pickle('data/df_enumP_para14gpext')
	program_list = df['program'].values

except FileNotFoundError:
	df = pd.read_pickle('data/df_enumP_para14')
	eR0 = df[df['progsize']==2]['eR'].max()
	dff = df[df['eR']>=eR0*.99] # prog at para14 with eR>eR0*.9999 covers all good prog at para80,146,212,278
	df = dff[['program', 'progsize']].to_pickle('data/df_enumP_para14gpext')
	program_list = df['program'].values


##%% FUNCTIONS
def gen_ao_list(seq_len=10):
	'''
	1m10s
	from ao2ao1ao0 to ao0ao1ao2
	'''
	ao_rev_list = [np.array(list(product([0,1], repeat=2*(i+1)))) for i in range(seq_len)]
	ao_id_list = [np.arange(2*(i+1)).reshape(i+1, 2)[::-1].flatten() for i in range(seq_len)]
	ao_list = [x[:,y] for x,y in zip(ao_rev_list, ao_id_list)]
	return ao_list, ao_rev_list

def gen_ao_idx_sort_dict(ao_list, seq_len=10):
	'''
	13s
	sort aoaoao such that (0,0,0,0,0,0) is on top while (1,1,1,1,1,1) is at the bottom
	'''
	ao_idx_sort_dict = {}
	for tau in range(seq_len):
		y = {tuple(x):i for i,x in enumerate(ao_list[tau])}
		idx_sort = np.array([y[k] for k in sorted(y)])
		ao_idx_sort_dict[tau] = idx_sort
	return ao_idx_sort_dict

def gen_Y_dict(Y, ao_list, ao_idx_sort_dict, prog_ids=[], seq_len=10):
	Y_dict = {}
	tau = seq_len-1
	idx_sort = ao_idx_sort_dict[tau] # need to sort for later rehape and sum
	Y = Y.todense()[idx_sort]
	#
	p_aoa_list = Y.reshape(2**(2*tau+1),2).sum(-1)
	p_aoa_dict = {tuple(x):y for x,y in zip(ao_list[tau][idx_sort][::2,:-1], p_aoa_list) if y>0}
	#
	p_ao_list = Y
	p_ao_dict = {tuple(x):y for x,y in zip(ao_list[tau][idx_sort], p_ao_list) if y>0}
	#
	Y_dict.update(p_aoa_dict)
	Y_dict.update(p_ao_dict)
	#
	aoaoa_list = [x for x in list(Y_dict.keys()) if (len(x)>1) & (len(x)%2==1)]
	return Y_dict, aoaoa_list

def aoaoa2wslg(aoaoa):
	aoa_aoa = [tuple(aoaoa[2*i:2*i+3]) for i in range(len(aoaoa)//2)]
	wslg_dict = {x:y for x,y in zip(list(product([0,1], repeat=3)), ['L','l','W','w','l','L','w','W'])}
	wslg = [wslg_dict[x] for x in aoa_aoa]
	return ''.join(wslg)

def gen_p_eR_dict(p, aoaoa_list):
	len_max = max([len(x) for x in aoaoa_list])
	aoaoa_lis = [x for x in aoaoa_list if len(x)==len_max]
	wslg_nz = [aoaoa2wslg(x) for x in aoaoa_lis]
	p_lis = [p[x] for x in aoaoa_lis]
	#
	p_dict = {x:y*2 for x,y in zip(wslg_nz, p_lis)}
	eR_dict = {x:(x.count('w')+x.count('W'))/len(x)*y for x,y in p_dict.items()}
	return p_dict, eR_dict

def gen_Y_sym_idx(depth):
	y0 = {tuple(fl(x)):i for i,x in enumerate(product([(1, -1), (1, 1), (-1, -1), (-1, 1)], repeat=depth))}
	y1 = {k:y0[k] for k in sorted(y0)}
	return list(y1.values())


##%% MP: both bdp & sdp
if False:
	# mp parameters
	dim_u = 180
	seq_len = 10
	Y_symm_idx = gen_Y_sym_idx(seq_len)
	ao_list, ao_rev_list = gen_ao_list(seq_len=seq_len)
	ao_idx_sort_dict = gen_ao_idx_sort_dict(ao_list, seq_len=seq_len)

	# mp functions
	def run(prog):
		# bdp & sdp
		eR, p_arr, t_iter = run_bdp(prog, para, dim_u=dim_u)
		Y = run_sdp(p_arr, prog, para, dim_u=dim_u, t_iter_max=seq_len)[-1]

		# symmetrize Y
		Y = (Y.todense() + Y.todense()[Y_symm_idx])/2
		Y = sparse.COO(Y)

		# to wslg
		p, aoaoa_list = gen_Y_dict(Y, ao_list, ao_idx_sort_dict, prog_ids=[], seq_len=seq_len)
		p_dict, eR_dict = gen_p_eR_dict(p, aoaoa_list)
		return eR, t_iter, p_arr, p_dict, eR_dict

	# run
	eR_, t_iter_, p_arr_, p_dict_, eR_dict_ = [[] for x in range(5)]
	with multiprocessing.Pool(num_cpus) as p:
		jobs = p.map(run, program_list)

		for k, job in enumerate(jobs):
			eR_.append(job[0])
			t_iter_.append(job[1])
			p_arr_.append(job[2])
			p_dict_.append(job[3])
			eR_dict_.append(job[4])

			if k % max(len(program_list)//100, 1)==0:
				print('program', k, job[0])

	# df
	dff = df[['progsize']]
	dff['eR'] = eR_
	dff['t_iter'] = t_iter_
	dff['p_arr'] = p_arr_
	dff['p_dict'] = p_dict_
	dff['eR_dict'] = eR_dict_
	dff.to_pickle('data/df_enumTaskP/df_enumTaskP_%s'%job_id)

	# df reduced; keey SDP only for h=.05 (enumP in other h has the same nz seq)
	dff_r = df[['progsize']]
	dff_r['eR'] = eR_
	dff_r['t_iter'] = t_iter_
	if job_id==1:
		dff_r['seq'] = [list(x.keys()) for x in p_dict_]
	dff_r.to_pickle('data/df_enumTaskP_r/df_enumTaskP_%s'%job_id)



# ##%% MP (2bd)
# if True:
# 	if not os.path.exists('data/df_enumTaskP_r'): os.makedirs('data/df_enumTaskP_r')
# 	for k, h in enumerate(h_list):
# 		job_id = k+1
# 		df = pd.read_pickle('data/df_enumTaskP/df_enumTaskP_%s'%job_id)
# 		dff = df[['progsize', 'eR', 't_iter']]
# 		if job_id==1:
# 			dff['seq'] = [list(x.keys()) for x in df['p_dict']]
# 		dff.to_pickle('data/df_enumTaskP_r/df_enumTaskP_%s'%job_id)
