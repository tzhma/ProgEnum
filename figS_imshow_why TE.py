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
from core_enum import fl
import community as community_louvain


## LOAD
df_pte = pd.read_pickle('data/df_gpn_pte')
pprog_arr = pickle.load(open('data/df_gpn_pprog_arr', 'rb'))
dprog_arr = pickle.load(open('data/df_gpn_dprog_arr', 'rb'))

dff = df_pte[df_pte['eR']>=.277911]
tar_list = df_pte['target'].values
itar = [tar_list.tolist().index(x) for x in dff['target']]
uni = dff['id_unique'].values


## PRINT: for selecting num_prog, histsize, histdensity & coasely tune to pp_sum
df14 = pd.read_pickle('data/df_enumTaskP_r/df_enumTaskP_1')
pp_sum = dff['pp'].sum()
mean_num_seq_mean = np.mean([len(x) for x in df14.loc[uni, 'seq']])
total_num_seq = len(np.unique(fl(df14.loc[uni, 'seq'].values)))
print(len(uni), mean_num_seq_mean, total_num_seq, pp_sum)
# print: (4230 1927.8891252955082 103770 30.78369481937469)
# use:	 (~4100 2000 100000, 30)


## PRINT: mean(# 1-mut) in GPN
dprog_arr_r = dprog_arr[itar,:][:,itar]
print((dprog_arr_r==1).sum(1).min(), (dprog_arr_r==1).sum(1).max(), (dprog_arr_r==1).sum(1).mean())


##%% RUN: community_louvain on 4230 good programs (3m)
# parameters
np.random.seed(42)
rho=.5

# load graph
pprog_arr_r = pprog_arr[itar,:,-1][:,itar]
G = nx.from_numpy_array(pprog_arr_r)

# community_louvain
cl_dict = community_louvain.best_partition(G, weight='weight', resolution=.5)
num_cl = max(list(cl_dict.values())) + 1
blocks = [[] for i in range(num_cl)]
for id_prog, id_blk in cl_dict.items(): blocks[id_blk].append(id_prog)
id_blk = fl(blocks)

# plot
title = len(uni), round(mean_num_seq_mean, 3), round(total_num_seq, 3), round(pp_sum, 3), 'num_cl=%s'%num_cl

plt.figure(figsize=(10,4), dpi=300)
plt.suptitle(title)
plt.subplot(121)
plt.imshow(pprog_arr_r[id_blk,:][:,id_blk], cmap='Greys', alpha=1, vmax=pprog_arr_r.mean()+pprog_arr_r.std())
plt.xlabel('sorted prog j')
plt.ylabel('sorted prog i')

plt.subplot(122)
pprog_arr_r.flatten()
plt.hist(pprog_arr_r.flatten(), bins=100, color='silver')
plt.yscale('log')
plt.xlabel('pp for all (i,j)')
plt.ylabel('count')

plt.savefig('fig/whyTE_gpn.svg')


##%% RUN: pp stat from 10 toy clusters
# parameters
np.random.seed(42)
num_prog = 4100
histsize = 100000
density = .02
num_p0 = 10

num_p1 = num_prog//num_p0
r0 = density * .99
r1_0 = density * .01
nnz = int(histsize * density)

# run
p0_0 = np.zeros(histsize)
p0_arr = np.zeros([num_p0, histsize])
for i in range(num_p0):
	p0_arr[i, i*nnz:(i+1)*nnz] = 1

p1_list = []
for p0 in p0_arr:
	r1 = r1_0 # np.random.exponential(r1_0)
	p1 = (p0 + np.random.choice([0,1], [num_p1, histsize], p=[1-r1,r1])) % 2
	p1_list.append(p1)
p1_arr = np.concatenate(p1_list, 0)
p1_arr = p1_arr/(p1_arr.sum(1)[:,None]+1e-16)
pp_arr = (p1_arr/(p1_arr.sum(0) + 1e-16)) @ p1_arr.T

# plot
title = len(p1_arr), round((p1_arr>0).sum(1).mean(), 3), round(pp_arr[range(len(pp_arr)),range(len(pp_arr))].sum(), 3)

plt.figure(figsize=(10,4), dpi=300)
plt.suptitle(title)
plt.subplot(121)
plt.imshow(pp_arr, cmap='Greys', vmax=pp_arr.mean()+pp_arr.std())
plt.xlabel('prog j')
plt.ylabel('prog i')

plt.subplot(122)
plt.hist(pp_arr.flatten(), bins=100, color='silver')
plt.yscale('log')
plt.xlabel('pp for all (i,j)')
plt.ylabel('count')

plt.savefig('fig/whyTE_toyClusters.svg')


##%% RUN: pp stat from a toy tree
# parameters
np.random.seed(42)
histsize = 100000
density = .02
num_mut = 2
num_gen = 11

r0 = density * .8
r1_0 = density * .2 / num_gen
nnz = int(histsize*density)

# run
p0 = np.zeros([1,histsize])
p0[:,:nnz] = 1
p0_lists = [[p0.tolist()]]
p1_lists = [p0]
for i in range(num_gen):
	p0_list = p0_lists[-1]
	p1_list = []
	for p0 in p0_list:
		r1 = r1_0 # np.random.exponential(r1_0)
		p1_lis = (np.array(p0) + np.random.choice([0,1], [num_mut, histsize], p=[1-r1,r1])) % 2
		p1_list += p1_lis.tolist()
		p1_lists.append(p1_lis)
	p0_lists.append(p1_list.copy())

p1_arr = np.concatenate(p1_lists, 0)
p1_arr = p1_arr/(p1_arr.sum(1)[:,None]+1e-16)
pp_arr = (p1_arr/(p1_arr.sum(0) + 1e-16)) @ p1_arr.T


# plot
title = len(p1_arr), round((p1_arr>0).sum(1).mean(), 3), round(pp_arr[range(len(pp_arr)),range(len(pp_arr))].sum(), 3)

plt.figure(figsize=(10,4), dpi=300)
plt.suptitle(title)
plt.subplot(121)
plt.imshow(pp_arr, cmap='Greys', vmax=pp_arr.mean()+pp_arr.std())
plt.xlabel('prog j')
plt.ylabel('prog i')

plt.subplot(122)
plt.hist(pp_arr.flatten(), bins=100, color='silver')
plt.yscale('log')
plt.xlabel('pp for all (i,j)')
plt.ylabel('count')

plt.savefig('fig/whyTE_toyTree.svg')
