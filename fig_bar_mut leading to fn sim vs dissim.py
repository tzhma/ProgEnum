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


## LOAD
df_pte = pd.read_pickle('data/df_gpn_pte')
df_seq = pd.read_pickle('data/df_gpn_seq_motif_1')
df_seq['motifs'] = [tuple(sorted(x)) for x in df_seq['motifs']]

tar_list = df_pte['target'].values
Y_lists = pickle.load(open('data/df_gpn_Y_lists','rb')) # 10 ao_hists for all programs
denom_list = pickle.load(open('data/denom_list','rb'))


## FUNCTIONS
def gen_ao_list(len_ao):
	'''
	1m10s
	from ao2ao1ao0 to ao0ao1ao2
	'''
	ao_rev_list = [np.array(list(product([0,1], repeat=2*(i+1)))) for i in range(len_ao)]
	ao_id_list = [np.arange(2*(i+1)).reshape(i+1, 2)[::-1].flatten() for i in range(len_ao)]
	ao_list = [x[:,y] for x,y in zip(ao_rev_list, ao_id_list)]
	return ao_list, ao_rev_list

def gen_ao_idx_sort_dict(ao_list, len_ao):
	'''
	13s
	sort aoaoao such that (0,0,0,0,0,0) is on top while (1,1,1,1,1,1) is at the bottom
	'''
	ao_idx_sort_dict = {}
	for tau in range(len_ao):
		y = {tuple(x):i for i,x in enumerate(ao_list[tau])}
		idx_sort = np.array([y[k] for k in sorted(y)])
		ao_idx_sort_dict[tau] = idx_sort
	return ao_idx_sort_dict

def gen_Y_dict(Y_list, ao_idx_sort_dict, len_ao, prog_ids=[]):
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
	for tau in range(len_ao):
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

		if tau==len_ao-1:
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

def gen_p_pp_dict(p, pp, aoaoa_list, len_ao):
	aoaoa_lis = [x for x in aoaoa_list if len(x)==len_ao*2-1]
	wslg_nz = [aoaoa2wslg(x) for x in aoaoa_lis]
	p_lis = [p[x] for x in aoaoa_lis]
	pp_lis = [pp[x] for x in aoaoa_lis]
	#
	p_dict = {x:y for x,y in zip(wslg_nz, p_lis)}
	pp_dict = {x:y for x,y in zip(wslg_nz, pp_lis)}
	return p_dict, pp_dict

def gen_wslg_p_list(tar, len_ao):
	Y0 = Y_lists[tar_list.tolist().index(tar)]
	p, pp, aoaoa_list = gen_Y_dict(Y0, ao_idx_sort_dict, len_ao)
	p_dict, pp_dict = gen_p_pp_dict(p, pp, aoaoa_list, len_ao) # key:wslg_nz
	wslg_nz = sorted(list(p_dict.keys()))
	# p_list, pp_list = np.array([p_dict[x] for x in wslg_nz]), np.array([pp_dict[x] for x in wslg_nz])
	return wslg_nz, p_dict, aoaoa_list


# ## FIND cand programs
# print(df_pte[df_pte['source']==5]['pp_sim'][:50])


##%% RUN
tar0, tar1, tar2 = 5, 7, 5407
len_ao = 4
ao_list, ao_rev_list = gen_ao_list(len_ao)
ao_idx_sort_dict = gen_ao_idx_sort_dict(ao_list, len_ao)

wslg0, p0, _ = gen_wslg_p_list(tar0, len_ao)
wslg1, p1, _ = gen_wslg_p_list(tar1, len_ao)
wslg2, p2, _ = gen_wslg_p_list(tar2, len_ao)
wslg_all = np.unique(wslg0 + wslg1 + wslg2)

p_arr = np.zeros([3, wslg_all.size])
for i, wslg in enumerate(wslg_all):
	try: p_arr[0,i] = p0[wslg]
	except KeyError: pass
	try: p_arr[1,i] = p1[wslg]
	except KeyError: pass
	try: p_arr[2,i] = p2[wslg]
	except KeyError: pass

idx_sort = np.argsort(p_arr[0] - p_arr[1]/100)[::-1]


## PLOT
plt.figure(figsize=(8,4), dpi=300)
plt.subplot(211)
plt.bar(np.arange(len(wslg_all))-.2, p_arr[0,idx_sort], color='#B3B3B3', label='prog %s'%tar0, width=.4)
plt.bar(np.arange(len(wslg_all))+.2, p_arr[1,idx_sort], color='#4A4A4A', label='prog %s'%tar1, width=.4)
plt.xticks([])
plt.yticks([])
plt.ylabel('p(seq)')
plt.ylim([0,.2])
plt.legend()

plt.subplot(212)
plt.bar(np.arange(len(wslg_all))-.2, p_arr[0,idx_sort], color='#B3B3B3', label='prog %s'%tar0, width=.4)
plt.bar(np.arange(len(wslg_all))+.2, p_arr[2,idx_sort], color='#4A4A4A', label='prog %s'%tar2, width=.4)
plt.xticks(range(len(wslg_all)), labels=np.array(wslg_all)[idx_sort], rotation='-45')
plt.yticks([])
plt.xlabel('sequence')
plt.ylabel('p(seq)')
plt.ylim([0,.2])
plt.legend()

plt.tight_layout()
plt.savefig('fig/seq-hist_5_7_5407.svg')
