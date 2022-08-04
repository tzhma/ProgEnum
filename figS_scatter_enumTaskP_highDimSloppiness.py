import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import os
import sys
from itertools import permutations, combinations, product
import pandas as pd
from core_bdpB import run_bdpB
from core_enum import fl
import multiprocess

num_cpus = int(multiprocess.cpu_count()*1)
print('num_cpus = %s'%num_cpus)


## PARAMETERS
dp, dpm = .3, -.5
h_list = np.around(np.linspace(.05, .32, 28), 6)[:-1].tolist() + np.around(np.linspace(.32, .4, 41), 6).tolist()

idx_coarse = np.arange(27).tolist() + (27+np.arange(41)[::5]).tolist()
idx_fine = np.arange(-41,-15)
h_list_c = np.array(h_list)[idx_coarse]
h_list_f = np.array(h_list)[idx_fine]

## critical h
h_c0 = 1/2 + (1-dpm)/(1+dp**2-dpm**2) - 4/(dp**2+(3-dpm)*(1+dpm))
h_c1 = 0.20317
h_c2 = -dpm/(1-dpm)
h_c3 = (1-dpm)/4 - dp**2/4/(1-dpm)

## LOAD
dfB = pd.read_pickle('data/df_enumTaskB')
dfB['seq'] = [list(x.keys()) for x in dfB['pB10']]
df14 = pd.read_pickle('data/df_enumTaskP_r/df_enumTaskP_1')
seq_WSLG = df14.loc[4, 'seq']


##%% FUNCTION
def gen_slop_vec(job_id):
	# get seq_B
	seq_B = dfB.loc[job_id-1, 'seq']

	# get seq_enumTaskP
	df = pd.read_pickle('data/df_enumTaskP_r/df_enumTaskP_%s'%job_id)
	eRWSLG = df[df['progsize']==2]['eR'].max()
	dff = df[df['eR']>=eRWSLG]
	# seq_enumTaskP = np.unique(fl(df14.loc[dff.index, 'seq'].values))
	seq_enumTaskP = fl(df14.loc[dff.index, 'seq'].values)

	#
	slop_vec = [sum([x.count(y) for x in seq_enumTaskP])+1 for y in ['L','W','l','w']]
	# slop_vec = [sum([(y in x)*1 for x in seq_enumTaskP]) for y in ['L','W','l','w']]
	return slop_vec


## MP (15s)
job_id_list = range(1,69)

slop_vec_ = []
with multiprocess.Pool(num_cpus) as p:
	jobs = p.map(gen_slop_vec, job_id_list)
	for k, job in enumerate(jobs):
		slop_vec_.append(job)
		print(k+1, job)


##%% PLOT
h_id = [20,25,30,35,40,45,50,55,60]

plt.figure(figsize=(10,4), dpi=300)
for i,j in enumerate(h_id):
	plt.axvline(h_list[j], color='lightgray', linestyle='--')
plt.semilogy(h_list[20:61], np.array(slop_vec_)[20:61,0], 'k--') # L
plt.semilogy(h_list[20:61], np.array(slop_vec_)[20:61,1], 'k') # W
plt.semilogy(h_list[20:61], np.array(slop_vec_)[20:61,2], 'k--') # l
plt.semilogy(h_list[20:61], np.array(slop_vec_)[20:61,3], 'k') # w
plt.xlabel('hazard rate')
plt.ylabel('count in GPN')
plt.savefig('fig/taskSloppiness_enumTaskP_highDimSlop_1.svg')

plt.figure(figsize=(10,2), dpi=300)
for i,j in enumerate(h_id):
	ax = plt.subplot(1,len(h_id),i+1, polar=True)
	theta = np.linspace(0, 2*np.pi, 5)
	r = np.log10(slop_vec_[j] + [slop_vec_[j][0]]) #2 + np.sin(theta * 2)
	ax.plot(theta, r, color='black', ls='-', linewidth=1)
	ax.fill(theta, r, 'tab:cyan', alpha=.5)
	# ax.set_xticks([0,np.pi/2,np.pi,np.pi/2*3], ['L','W','l','w'])
	ax.set_xticks([0,np.pi/2,np.pi,np.pi/2*3], ['','','',''])
	ax.set_yticks([])
	ax.set_ylim([0,8])
plt.savefig('fig/taskSloppiness_enumTaskP_highDimSlop_2.svg')
