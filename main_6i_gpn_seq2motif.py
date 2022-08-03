'''
ppc_thres=.95, 30m on cluster
ppc_thres=.99, 3h on cluster
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
from core_sdp import run_sdp, gen_pprog_lists, gen_denom_list
from core_enum import fl, reshape_prog, gen_stdz_prog, partition
import multiprocessing

num_cpus = int(multiprocessing.cpu_count()*1)
print('num_cpus = %s'%num_cpus)


## LOAD
df_pte = pd.read_pickle('data/df_gpn_pte')
tar_list = df_pte['target'].values
Y_lists = pickle.load(open('data/df_gpn_Y_lists','rb')) # 10 ao_hists for all programs
denom_list = pickle.load(open('data/denom_list','rb'))


## FUNCTIONS
def gen_ao_list():
	'''
	1m10s
	from ao2ao1ao0 to ao0ao1ao2
	'''
	ao_rev_list = [np.array(list(product([0,1], repeat=2*(i+1)))) for i in range(10)]
	ao_id_list = [np.arange(2*(i+1)).reshape(i+1, 2)[::-1].flatten() for i in range(10)]
	ao_list = [x[:,y] for x,y in zip(ao_rev_list, ao_id_list)]
	return ao_list, ao_rev_list

def gen_ao_idx_sort_dict(ao_list):
	'''
	13s
	sort aoaoao such that (0,0,0,0,0,0) is on top while (1,1,1,1,1,1) is at the bottom
	'''
	ao_idx_sort_dict = {}
	for tau in range(10):
		y = {tuple(x):i for i,x in enumerate(ao_list[tau])}
		idx_sort = np.array([y[k] for k in sorted(y)])
		ao_idx_sort_dict[tau] = idx_sort
	return ao_idx_sort_dict

def gen_Y_dict(Y_list, ao_idx_sort_dict, prog_ids=[]):
	'''
	2s for global pp; 3s for local pp
	prog_ids=[] for global pp
	for the cases of local or pair pp, denom_list is re-computed among prog_ids
	'''
	# override denom_list
	if len(prog_ids)!=0: denom_lis = [np.stack(Y_lists[tar_list.tolist().index(tar)][tau].todense() for tar in prog_ids).sum(0) for tau in range(10)]
	else: denom_lis = denom_list.copy()

	Y_dict = {}
	pp_dict = {}
	for tau in range(10):
		idx_sort = ao_idx_sort_dict[tau] # need to sort for later rehape and sum
		Y = Y_list[tau].todense()[idx_sort]
		#
		p_aoa_list = Y.reshape(2**(2*tau+1),2).sum(-1)
		p_aoa_dict = {tuple(x):y for x,y in zip(ao_list[tau][idx_sort][::2,:-1], p_aoa_list) if y>0}
		#
		p_ao_list = Y
		p_ao_dict = {tuple(x):y for x,y in zip(ao_list[tau][idx_sort], p_ao_list) if y>0}
		#
		Y_dict.update(p_aoa_dict)
		Y_dict.update(p_ao_dict)

		if tau==9:
			denom_r = denom_lis[tau][idx_sort].reshape(2**(2*tau+1),2).sum(-1)
			Y_r = p_aoa_list
			pp_aoa_list = (Y_r/denom_r) * Y_r
			pp_aoa_dict = {tuple(x):y for x,y in zip(ao_list[tau][idx_sort][::2,:-1], pp_aoa_list) if y>0}
			pp_dict.update(pp_aoa_dict)
	#
	aoaoa_list = [x for x in list(Y_dict.keys()) if (len(x)>1) & (len(x)%2==1)]
	return Y_dict, pp_dict, aoaoa_list

def aoaoa2wslg(aoaoa):
	aoa_aoa = [tuple(aoaoa[2*i:2*i+3]) for i in range(len(aoaoa)//2)]
	wslg_dict = {x:y for x,y in zip(list(product([0,1], repeat=3)), ['L','l','W','w','l','L','w','W'])}
	wslg = [wslg_dict[x] for x in aoa_aoa]
	return ''.join(wslg)

def gen_p_pp_dict(p, pp, aoaoa_list):
	aoaoa_lis = [x for x in aoaoa_list if len(x)==19]
	wslg_nz = [aoaoa2wslg(x) for x in aoaoa_lis]
	p_lis = [p[x] for x in aoaoa_lis]
	pp_lis = [pp[x] for x in aoaoa_lis]
	#
	p_dict = {x:y for x,y in zip(wslg_nz, p_lis)}
	pp_dict = {x:y for x,y in zip(wslg_nz, pp_lis)}
	return p_dict, pp_dict

def gen_wslg_nz(aoaoa_list):
	aoaoa_lis = [x for x in aoaoa_list if len(x)==19]
	wslg_nz = [aoaoa2wslg(x) for x in aoaoa_lis]
	return wslg_nz

def gen_pp_coverage(seqs, motifs, p50, pp50):
	occupied = []
	diff_covered = []
	for seq in seqs:
		occ = []
		dc = []
		for m in motifs:
			idx = fl([range(i,i+len(m)) for i,x in enumerate(seq) if m in seq[i:i+len(m)]])
			occ += idx
			occ = sorted(set(occ))
			if len(dc)==0: _dc = len(occ)
			else: _dc = len(occ) - sum(dc)
			dc.append(_dc)
		# occupied.append(sorted(set(occ)))
		diff_covered.append(dc)
	#
	pm_arr = np.array(diff_covered)/9
	p50_arr = pm_arr * p50[:,None]
	pp50_arr = pm_arr * pp50[:,None]
	#
	pp_coverage = pp50_arr.sum()/pp50.sum()
	p_coverage = p50_arr.sum()/p50.sum()
	return pp_coverage, p_coverage, pm_arr, p50_arr, pp50_arr

def gen_seqs_motifs_p50_pp50(tar, prog_ids):
	'''
	'''
	# load
	Y_list = Y_lists[tar_list.tolist().index(tar)]
	p, pp, aoaoa_list = gen_Y_dict(Y_list, ao_idx_sort_dict, prog_ids=prog_ids)
	p_dict, pp_dict = gen_p_pp_dict(p, pp, aoaoa_list) # key:wslg_nz
	wslg_nz = sorted(list(p_dict.keys()))
	p_list, pp_list = np.array([p_dict[x] for x in wslg_nz]), np.array([pp_dict[x] for x in wslg_nz])

	# seq50, p50, pp50
	num_top50 = np.where((np.sort(pp_list)[::-1].cumsum() - pp_list.sum()/2)>0)[0][0]
	idx_sort = pp_list.argsort()[::-1][:num_top50]
	seqs = [wslg_nz[x] for x in idx_sort]
	pp50 = pp_list[idx_sort]
	p50 = p_list[idx_sort]

	# get loop cand
	loop_cand = []
	for wslg in wslg_nz:
		loop_cand += fl([[wslg[i:i+l] for i in range(10-l)] for l in range(1,10)])
	loop_cand = sorted(set(loop_cand))

	# loop pass
	loop_list = []
	for loop in loop_cand:
		loop_sl = [loop[i:]+loop[:i] for i in range(len(loop))]
		seq_from_loop = [(x*9)[:9] for x in loop_sl]
		if np.prod([x in wslg_nz for x in seq_from_loop])==1:
			loop_list.append(loop)

	# loop_uni
	loop_uni_list = []
	loop_sl_list = []
	# loop_id2loop_uni = {}
	for i, loop in enumerate(loop_list):
		loop_sl = [loop[i:]+loop[:i] for i in range(len(loop))]
		loop_uni = [x for x in loop_sl if x in loop_uni_list]
		if len(loop_uni)>0: continue
		else:
			loop_uni_list.append(loop)
			loop_sl_list.append(sorted(set(loop_sl)))

	idx_sort = np.argsort([len(x) for x in loop_uni_list])
	loop_uni_sort = [loop_uni_list[x] for x in idx_sort]
	loop_sl_sort = [loop_sl_list[x] for x in idx_sort]
	loop_sl_fl = fl(loop_sl_sort)
	tuples = fl([[(x,z) for z in y] for x,y in zip(loop_uni_sort, loop_sl_sort)])
	loop_sl_dict = {y:x for x,y in tuples}
	return seqs, loop_sl_fl, loop_sl_dict, p50, pp50

def gen_df50(tar, seqs, loop_sl_fl, loop_sl_dict, p50, pp50):
	'''
	'''
	ppc_thres = .95
	ppc_max, _, _, _, _ = gen_pp_coverage(seqs, loop_sl_fl, p50, pp50)

	#
	m_groups = [[x for x in loop_sl_fl if len(x)==y] for y in range(1,10)]
	motifs = []
	motifs_reject = []
	pp_cumsum = [0]
	break_flag = False
	for m_group in m_groups:
		if break_flag: break

		for mm in m_group: # loop k times to finish m_group
			motifs_checked = motifs + motifs_reject
			m_remain = [x for x in m_group if x not in motifs_checked]
			ppc = []
			for m in m_remain:
				motifs_test = motifs + [m]
				pp_coverage, p_coverage, pm_arr, p50_arr, pp50_arr = gen_pp_coverage(seqs, motifs_test, p50, pp50)
				ppc.append(pp_coverage)

			# skip if new motif explain nothing
			if (max(ppc) - pp_cumsum[-1])<1e-6:
				motifs_reject.append(m)
				continue

			# append new motif
			idx_max = np.argmax(ppc)
			motifs.append(m_remain[idx_max])
			pp_cumsum.append(ppc[idx_max])

			# break if motifs explain 90% ppc_max
			if (pp_cumsum[-1]/ppc_max)>ppc_thres:
				break_flag = True
				break

	# compute final pp_coverage
	_, _, pm_arr, p50_arr, pp50_arr = gen_pp_coverage(seqs, motifs, p50, pp50)
	ppm_cumsum = pp_cumsum[1:] + [ppc_max]

	# collapse to motifs_uni
	motifs_uni = sorted(set([loop_sl_dict[x] for x in motifs]), key=len)
	m2id = {x:i for i,x in enumerate(motifs_uni)}
	id2uni = {i:m2id[loop_sl_dict[x]] for i,x in enumerate(motifs)}
	#
	p50_uni = np.zeros([len(seqs), len(motifs_uni)])
	pp50_uni = np.zeros([len(seqs), len(motifs_uni)])
	for i, uni in id2uni.items():
		p50_uni[:,uni] += p50_arr[:,i]
		pp50_uni[:,uni] += pp50_arr[:,i]

	# df_arr
	df_arr = pd.DataFrame([[tar, seqs, motifs_uni, pm_arr, p50_uni, pp50_uni, p50, pp50, ppm_cumsum]], columns=['prog_id', 'seqs', 'motifs', 'pm_s', 'psm', 'ppsm', 'ps', 'pps', 'ppm_cumsum'])

	# df_seq
	m_lis = [tuple(np.array(motifs_uni)[np.where(x>0)]) for x in pp50_uni]
	ppm_lis = [x[np.where(x>0)] for x in pp50_uni]
	pm_lis = [x[np.where(x>0)] for x in p50_uni]
	ppm_sum = [x.sum() for x in ppm_lis]
	pm_sum = [x.sum() for x in pm_lis]
	df_seq = pd.DataFrame([seqs, m_lis, ppm_lis, pm_lis, pp50, p50, ppm_sum, pm_sum]).T
	df_seq.columns = ['seq', 'motifs', 'ppm_lis', 'pm_lis', 'pp', 'p', 'ppm_sum', 'pm_sum']
	df_seq['prog_id'] = tar

	# df_motif
	s_lis = [tuple(np.array(seqs)[np.where(x>0)]) for x in pp50_uni.T]
	pps_lis = [x[np.where(x>0)] for x in pp50_uni.T]
	ps_lis = [x[np.where(x>0)] for x in p50_uni.T]
	pps_sum = [x.sum() for x in pps_lis]
	ps_sum = [x.sum() for x in ps_lis]
	df_motif = pd.DataFrame([motifs_uni, s_lis, pps_lis, ps_lis, pps_sum, ps_sum]).T
	df_motif.columns = ['motif', 'seqs', 'pps_lis', 'ps_lis', 'pps_sum', 'ps_sum']
	df_motif['prog_id'] = tar

	return df_arr, df_seq, df_motif

def gen_df50_4levels(tar):
	'''
	...
	'''
	# get prog_id_lists (pair, local, global)
	parent = df_pte.loc[tar, 'source']
	siblings = df_pte[df_pte['source']==parent].index.values.tolist()
	dependents = df_pte[df_pte['source']==tar].index.values.tolist()
	_pair = [parent, tar]
	_local = [parent, tar] + siblings
	_parent = [tar] + dependents
	_global = []
	prog_id_lists = [_pair, _local, _parent, _global]
	labels = ['pair', 'local', 'parent', 'global']

	# get df
	df_arr_list, df_seq_list, df_motif_list = [], [], []
	for label, prog_ids in zip(labels, prog_id_lists):
		seqs, loop_sl_fl, loop_sl_dict, p50, pp50 = gen_seqs_motifs_p50_pp50(tar, prog_ids)
		df_arr, df_seq, df_motif = gen_df50(tar, seqs, loop_sl_fl, loop_sl_dict, p50, pp50)
		#
		df_arr['level'] = label
		df_seq['level'] = label
		df_motif['level'] = label

		# append
		df_arr_list.append(df_arr)
		df_seq_list.append(df_seq)
		df_motif_list.append(df_motif)

	# concat df
	df_arr_concat = pd.concat(df_arr_list, ignore_index=True)
	df_arr_concat.index = range(len(df_arr_concat))

	df_seq_concat = pd.concat(df_seq_list, ignore_index=True)
	df_seq_concat.index = range(len(df_seq_concat))

	df_motif_concat = pd.concat(df_motif_list, ignore_index=True)
	df_motif_concat.index = range(len(df_motif_concat))
	return df_arr_concat, df_seq_concat, df_motif_concat


## MP
if False:
	ao_list, ao_rev_list = gen_ao_list()
	ao_idx_sort_dict = gen_ao_idx_sort_dict(ao_list)

	df_arr_list, df_seq_list, df_motif_list = [], [], []
	with multiprocessing.Pool(num_cpus) as p:
		jobs = p.map(gen_df50_4levels, tar_list)
		for tar, job in zip(tar_list, jobs):
			df_arr_list.append(job[0])
			df_seq_list.append(job[1])
			df_motif_list.append(job[2])

	# assemble
	df_arr = pd.concat(df_arr_list, ignore_index=True)
	df_arr.index = range(len(df_arr))
	pickle.dump(df_arr, open('data/df_gpn_seq_motif_0', 'wb'))

	df_seq = pd.concat(df_seq_list, ignore_index=True)
	df_seq.index = range(len(df_seq))
	pickle.dump(df_seq, open('data/df_gpn_seq_motif_1', 'wb'))

	df_motif = pd.concat(df_motif_list, ignore_index=True)
	df_motif.index = range(len(df_motif))
	pickle.dump(df_motif, open('data/df_gpn_seq_motif_2', 'wb'))


# ##%% TEST: prep 14s
# ao_list, ao_rev_list = gen_ao_list()
# ao_idx_sort_dict = gen_ao_idx_sort_dict(ao_list)
#
# ##%% TEST: single (18s)
# df_arr, df_seq, df_motif = gen_df50_4levels(0)
# df_motif[(df_motif['level']=='parent')]
