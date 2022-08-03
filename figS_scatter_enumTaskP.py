'''
...
performance_min, performance_max = 1.000, 0.069
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import pickle
import time
import os
import sys
from itertools import permutations, combinations, product
import pandas as pd
from core_bdpB import run_bdpB


## PARAMETERS
dp, dpm = .3, -.5
h_list = np.around(np.linspace(.05, .32, 28), 6)[:-1].tolist() + np.around(np.linspace(.32, .4, 41), 6).tolist()


## critical h
h_c0 = 1/2 + (1-dpm)/(1+dp**2-dpm**2) - 4/(dp**2+(3-dpm)*(1+dpm))
h_c1 = 0.20317
h_c2 = -dpm/(1-dpm)
h_c3 = (1-dpm)/4 - dp**2/4/(1-dpm)


## FUNCTIONS
def gen_enumTaskP_data():
	'''
	13s
	'''
	## PARAMETERS
	dp, dpm = .3, -.5
	h_list = np.around(np.linspace(.05, .32, 28), 6)[:-1].tolist() + np.around(np.linspace(.32, .4, 41), 6).tolist()
	h_list = np.array(h_list)

	## extract fine-, coarse-grained sweeps
	idx_coarse = np.arange(27).tolist() + (27+np.arange(41)[::5]).tolist()
	idx_fine = np.arange(-41,-15)

	## extract num_prog, eR from dfs
	num_prog_list = []
	eR_max_list = []
	for k, h in enumerate(h_list):
		job_id = k+1
		if job_id==0: df = pd.read_pickle('data/df_enumP_para14')
		else: df = pd.read_pickle('data/df_enumTaskP_r/df_enumTaskP_%s'%job_id)
		eR0 = df[df['progsize']==2]['eR'].max()

		df_lis = [df[(df['progsize']==x) & (df['eR']>=eR0)] for x in [2,3,4,5]]
		num_prog = [len(x) for x in df_lis]
		eR_max = [x['eR'].max() for x in df_lis]
		num_prog_list.append(num_prog)
		eR_max_list.append(eR_max)

		print(h, num_prog, eR_max)

	num_prog_arr = np.array(num_prog_list).T
	eR_max_arr = np.array(eR_max_list).T

	## run_bdpB for all h (11s)
	eR_B_list = []
	p_arr_B_list = []
	for h in h_list:
		para = h, dp, dpm
		eR_B, p_arr, _ = run_bdpB(para)
		eR_B_list.append(eR_B)
		p_arr_B_list.append(p_arr)
	eR_B_list = np.array(eR_B_list)

	## from eR to performance
	performance_max_arr = (eR_max_arr-eR_max_arr[0])/(eR_B_list-eR_max_arr[0]+1e-16)
	print(np.array([x for x in performance_max_arr.flatten() if x>0]).max(), np.array([x for x in performance_max_arr.flatten() if x>0]).min())

	return h_list, num_prog_arr, eR_max_arr, eR_B_list, performance_max_arr, idx_coarse, idx_fine


##%% RUN
h_list, num_prog_arr, eR_max_arr, eR_B_list, performance_max_arr, idx_coarse, idx_fine = gen_enumTaskP_data()


##%% PLOT
size_arr = performance_max_arr*80 + 1
colors = [cm.get_cmap('tab20b')(i) for i in [3,2,1,0]]

plt.figure(figsize=(10,8), dpi=300)
plt.subplot(211)
plt.scatter(h_list[idx_coarse], num_prog_arr.T[idx_coarse,0], s=size_arr.T[idx_coarse,0], label='M=2', color=colors[0])
plt.scatter(h_list[idx_coarse], num_prog_arr.T[idx_coarse,1], s=size_arr.T[idx_coarse,1], label='M=3', color=colors[1])
plt.scatter(h_list[idx_coarse], num_prog_arr.T[idx_coarse,2], s=size_arr.T[idx_coarse,2], label='M=4', color=colors[2])
plt.scatter(h_list[idx_coarse], num_prog_arr.T[idx_coarse,3], s=size_arr.T[idx_coarse,3], label='M=5', color=colors[3])
plt.axvline(x=h_c0, color='deeppink', linestyle='--', linewidth=1)
plt.axvline(x=h_c1, color='deeppink', linestyle='--', linewidth=1)
plt.axvline(x=h_c2, color='deeppink', linestyle='--', linewidth=1)
plt.axvline(x=h_c3, color='deeppink', linestyle='--', linewidth=1)
plt.yscale('log')
plt.legend()
plt.xlabel('h')
plt.ylabel('# programs')

plt.subplot(212)
plt.scatter(h_list[idx_fine], num_prog_arr.T[idx_fine,0], s=size_arr.T[idx_fine,0], label='M=2', color=colors[0])
plt.scatter(h_list[idx_fine], num_prog_arr.T[idx_fine,1], s=size_arr.T[idx_fine,1], label='M=3', color=colors[1])
plt.scatter(h_list[idx_fine], num_prog_arr.T[idx_fine,2], s=size_arr.T[idx_fine,2], label='M=4', color=colors[2])
plt.scatter(h_list[idx_fine], num_prog_arr.T[idx_fine,3], s=size_arr.T[idx_fine,3], label='M=5', color=colors[3])
plt.axvline(x=h_c2, color='deeppink', linestyle='--', linewidth=1)
plt.axvline(x=h_c3, color='deeppink', linestyle='--', linewidth=1)
plt.yscale('log')
plt.legend()
plt.xlabel('h')
plt.ylabel('# programs')

plt.savefig('fig/taskSloppiness_enumTaskP.svg')


#%%
# pd.read_pickle('data/df_enumP_para14gpext')
# dff = df14.sort_values('eR', ascending=False)
# plt.plot(dff[(dff['eR']<=eR14) & (dff['eR']>=eR14*.99)]['eR'].values/eR14)
# plt.axhline(y=.99, color='r', linestyle='--')
# print([num_prog_arr[i][h_list.tolist().index(.33)] for i in range(4)])
