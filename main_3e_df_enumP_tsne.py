'''
2h on cluster
wt: 77h
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import pickle
import os
import sys
import warnings
warnings.filterwarnings('ignore')
import time
from itertools import permutations, combinations, product, islice
import pandas as pd
from core_enum import fl
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import multiprocess

num_cpus = int(multiprocess.cpu_count()*1)
print('num_cpus = %s'%num_cpus)


## LOAD
df = pd.read_pickle('data/df_enumP_para14')


## FUNCTIONS
def gen_inv_perm_list(outmap, keep_unpermuted=True, keep_flipped=True):
	'''
	generate permutations that
	1) doesn't change outmap
	2) change outmap to -outmap (task is symmetric under policy-inversion)
	'''
	outmap = list(outmap)
	y0 = []
	for perm in list(permutations(range(len(outmap)), len(outmap))):
		y1 = np.array(outmap)[list(perm)].tolist()
		if y1==outmap: y0.append(list(perm))
		elif (len(outmap)%2==0) & (y1==[-x for x in outmap]):
			if keep_flipped: y0.append(list(perm))
		else: pass
	if keep_unpermuted: return y0
	return y0[1:]

def gen_inmap_perm_list(program, perm_list):
	outmap, inmap = program
	dim_o = 2
	dim_m = len(inmap)
	y0 = inmap.copy()
	y1 = np.zeros([dim_m,2,dim_m])
	for io in range(dim_o): y1[range(dim_m),io,y0[:,io]] = 1
	y2 = []
	for perm in perm_list:
		y1_perm = y1[perm,:,:][:,:,perm] # permuted binary array
		idx_perm = np.where(y1_perm==1)
		y0_perm = y0.copy() # back to dim_mx2 table
		y0_perm[idx_perm[0],idx_perm[1]] = idx_perm[2]
		y2.append(y0_perm)
	return y2

def gen_struc_mat(program):
	dim_m_max = 5
	outmap, inmap = program
	dim_a, dim_o, dim_m = 2, 2, len(outmap)

	# find inmap with min struc_complx
	perm_list = gen_inv_perm_list(outmap)
	inmap_list = gen_inmap_perm_list(program, perm_list)
	inmap_zero = np.array([range(dim_m),range(dim_m)]).T
	str_cplx_lis = [np.sum(np.abs(x - inmap_zero)) for x in inmap_list]
	inmap_min = inmap_list[np.argmin(str_cplx_lis)]

	# compute struc_mat (win,stay,dis)
	y0 = np.zeros([dim_o,dim_a,dim_m_max])
	for (m,o) in list(product(range(dim_m),range(dim_o))):
		a = (outmap[m]>0)*1
		m1 = inmap_min[(m,o)]
		a1 = (outmap[m1]>0)*1
		# win = o
		# stay = int((a1==a)*1)
		# dis = int(np.abs(m1-m))
		y0[o,int((a1==a)*1),int(np.abs(m1-m))] += 1 # idx = (win,stay,dis)
	return y0


##%% MP: gen_df_enumP_struc_mat (32s)
program_list = df['program'].values
struc_mat_list = []
with multiprocess.Pool(num_cpus) as p:
	jobs = p.map(gen_struc_mat, program_list)

	for k, job in enumerate(jobs):
		struc_mat_list.append(job)

		if k % max(len(program_list)//100, 1)==0:
			print('prog', k, job)


##%% RUN: TSNE (5m)
w_lis = np.array([1,1,1,1,1])[None,None,:]
stru_mat_lis = [(x*w_lis).reshape(20) for x in df['struc_mat'].values]
stru_mat_array = np.stack(stru_mat_lis)

tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
vec = tsne.fit_transform(stru_mat_array)
tsne_x_list, tsne_y_list = vec[:,0], vec[:,1]


#%% df
df['struc_mat'] = struc_mat_list
df['tsne_x'] = tsne_x_list
df['tsne_y'] = tsne_y_list
df.to_pickle('data/df_enumP_para14_tsne')
