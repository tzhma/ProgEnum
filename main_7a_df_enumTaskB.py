'''
...
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
from core_bdp import run_bdp
from core_sdp import run_sdp
from core_bdpB import run_bdpB
from core_sdpB import run_sdpB
from core_enum import fl
import multiprocess

num_cpus = int(multiprocess.cpu_count()*1)
print('num_cpus = %s'%num_cpus)


## LOAD
dp, dpm = .3, -.5
h_list = np.around(np.linspace(.05, .32, 28), 6)[:-1].tolist() + np.around(np.linspace(.32, .4, 41), 6).tolist()
df_enumP = pd.read_pickle('data/df_enumP')
progWSLG = df_enumP.loc[4, 'program']


## FUNCTIONS
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
	# for tau in range(seq_len):
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


##%% MP: YB, YWSLG (15s)
if True:
	def run_mp1(h):
		para = h, dp, dpm
		dim_u = 180
		u_ub = ((1-2*h) + ((1+dpm)/dp*h)**2)**.5 - (1+dpm)/dp*h
		u_list = [-u_ub + 2*u_ub*i/(dim_u-2-1) for i in range(dim_u-2)]
		u0, u1 = u_list[0] - np.diff(u_list).mean(), u_list[-1] + np.diff(u_list).mean()
		u_list = [u0] + u_list + [u1]
		#
		eRB, pB_arr, _ = run_bdpB(para, dim_u=dim_u)
		YB2 = run_sdpB(pB_arr, para, dim_u=dim_u, t_iter_max=2)[-1]
		YB10 = run_sdpB(pB_arr, para, dim_u=dim_u, t_iter_max=10)[-1]
		#
		eRWSLG, pWSLG_arr, _ = run_bdp(progWSLG, para, dim_u=dim_u)
		YWSLG2 = run_sdp(pWSLG_arr, progWSLG, para, dim_u=dim_u, t_iter_max=2)[-1]
		YWSLG10 = run_sdp(pWSLG_arr, progWSLG, para, dim_u=dim_u, t_iter_max=10)[-1]
		return eRB, pB_arr, YB2, YB10, eRWSLG, pWSLG_arr, YWSLG2, YWSLG10, u_list

	# run
	eRB_, pB_arr_, YB2_, YB10_, eRWSLG_, pWSLG_arr_, YWSLG2_, YWSLG10_, u_list_ = [[] for i in range(9)]
	with multiprocess.Pool(num_cpus) as p:
		jobs = p.map(run_mp1, h_list)

		for i, job in enumerate(jobs):
			eRB_.append(job[0])
			pB_arr_.append(job[1])
			YB2_.append(job[2])
			YB10_.append(job[3])
			eRWSLG_.append(job[4])
			pWSLG_arr_.append(job[5])
			YWSLG2_.append(job[6])
			YWSLG10_.append(job[7])
			u_list_.append(job[8])

			print(h_list[i], job[0], job[4])

	# attr
	pp2_list = [(Y*Y0/(Y+Y0+1e-16)).sum() for Y, Y0 in zip(YB2_, YWSLG2_)]
	pp10_list = [(Y*Y0/(Y+Y0+1e-16)).sum() for Y, Y0 in zip(YB10_, YWSLG10_)]

	# df
	df = pd.DataFrame([h_list, u_list_, eRB_, pB_arr_, YB2_, YB10_, eRWSLG_, pWSLG_arr_, YWSLG2_, YWSLG10_]).T
	df.columns = ['h', 'u_list', 'eRB', 'pB_arr', 'YB2', 'YB10', 'eRWSLG', 'pWSLG_arr', 'YWSLG2', 'YWSLG10']
	df['pp2'] = pp2_list
	df['pp10'] = pp10_list


##%% MP: p(W), p(L), p(l) (2s)
if True:
	def run_mp2(h):
		seq_len = 2
		ao_list, ao_rev_list = gen_ao_list(seq_len=seq_len)
		ao_idx_sort_dict = gen_ao_idx_sort_dict(ao_list, seq_len=seq_len)

		# YB_list = [df[df['h']==h]['YB2'].values[0]]*2 # dummy *2
		YB = df[df['h']==h]['YB2'].values[0]
		pB, aoaoa_list = gen_Y_dict(YB, ao_list, ao_idx_sort_dict, prog_ids=[], seq_len=seq_len)
		pB_dict, eRB_dict = gen_p_eR_dict(pB, aoaoa_list)

		# Y_list = [df[df['h']==h]['YWSLG2'].values[0]]*2
		Y = df[df['h']==h]['YWSLG2'].values[0]
		p, aoaoa_list = gen_Y_dict(Y, ao_list, ao_idx_sort_dict, prog_ids=[], seq_len=seq_len)
		p_dict, eR_dict = gen_p_eR_dict(p, aoaoa_list)
		return pB_dict, eRB_dict, p_dict, eR_dict

	# run
	pB_dicts, eRB_dicts = [], []
	p_dicts, eR_dicts = [], []
	with multiprocess.Pool(num_cpus) as p:
		jobs = p.map(run_mp2, h_list)

		for i, job in enumerate(jobs):
			pB_dicts.append(job[0])
			eRB_dicts.append(job[1])
			p_dicts.append(job[2])
			eR_dicts.append(job[3])

			print(h_list[i], job[0])

	# attr
	ratio_pLpl_list = []
	for x in pB_dicts:
		try: ratio = x['L']/x['l']
		except KeyError: ratio = 0
		ratio_pLpl_list.append(ratio)

	# df
	df['pWSLG_WLl'] = p_dicts
	df['eRWSLG_WLl'] = eR_dicts
	df['pB_WLl'] = pB_dicts
	df['eRB_WLl'] = eRB_dicts
	df['ratio_pLpl'] = ratio_pLpl_list

	# pickle
	# df.to_pickle('data/df_enumTaskB')


##%% MP: p(seq) (4m)
if True:
	def run_mp3(h):
		seq_len = 10
		ao_list, ao_rev_list = gen_ao_list(seq_len=seq_len)
		ao_idx_sort_dict = gen_ao_idx_sort_dict(ao_list, seq_len=seq_len)

		YB = df[df['h']==h]['YB10'].values[0]
		pB, aoaoa_list = gen_Y_dict(YB, ao_list, ao_idx_sort_dict, prog_ids=[], seq_len=seq_len)
		pB_dict, eRB_dict = gen_p_eR_dict(pB, aoaoa_list)

		Y = df[df['h']==h]['YWSLG10'].values[0]
		p, aoaoa_list = gen_Y_dict(Y, ao_list, ao_idx_sort_dict, prog_ids=[], seq_len=seq_len)
		p_dict, eR_dict = gen_p_eR_dict(p, aoaoa_list)
		return pB_dict, eRB_dict, p_dict, eR_dict

	# run
	pB_dicts, eRB_dicts = [], []
	p_dicts, eR_dicts = [], []
	with multiprocess.Pool(num_cpus) as p:
		jobs = p.map(run_mp3, h_list)

		for i, job in enumerate(jobs):
			pB_dicts.append(job[0])
			eRB_dicts.append(job[1])
			p_dicts.append(job[2])
			eR_dicts.append(job[3])

			print(h_list[i], job[0])

	# df
	df['pWSLG10'] = p_dicts
	df['eRWSLG10'] = eR_dicts
	df['pB10'] = pB_dicts
	df['eRB10'] = eRB_dicts

	# pickle
	df.to_pickle('data/df_enumTaskB')


##%% MP: run_bdpB with special h
h_list = [.05, 0.111801, 0.112, .12, .13, .16, .20317, .28, .333333, .35, .36]

if False:
	def run_mp4(h):
		para = h, dp, dpm
		dim_u = 180
		u_ub = ((1-2*h) + ((1+dpm)/dp*h)**2)**.5 - (1+dpm)/dp*h
		u_list = [-u_ub + 2*u_ub*i/(dim_u-2-1) for i in range(dim_u-2)]
		u0, u1 = u_list[0] - np.diff(u_list).mean(), u_list[-1] + np.diff(u_list).mean()
		u_list = [u0] + u_list + [u1]
		#
		eRB, pB_arr, _ = run_bdpB(para, dim_u=dim_u)
		return eRB, pB_arr, u_list

	# run
	eRB_list, pB_arr_list, u_lists = [], [], []
	with multiprocess.Pool(num_cpus) as p:
		jobs = p.map(run_mp4, h_list)

		for i, job in enumerate(jobs):
			eRB_list.append(job[0])
			pB_arr_list.append(job[1])
			u_lists.append(job[2])

	# df
	df = pd.DataFrame([h_list, u_lists, eRB_list, pB_arr_list]).T
	df.columns = ['h', 'u_list', 'eRB', 'pB_arr']

	# pickle
	df.to_pickle('data/df_enumTaskB_special_h')
