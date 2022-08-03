'''
4m
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
from scipy.special import comb


## LOAD (30s)
df_pte = pd.read_pickle('data/df_gpn_pte')
df_arr = pd.read_pickle('data/df_gpn_seq_motif_0')
df_seq = pd.read_pickle('data/df_gpn_seq_motif_1')
df_motif = pd.read_pickle('data/df_gpn_seq_motif_2')


## FUNCTION
def gen_motif_corr(motifs, tuples):
	# motit_arr
	m2i = {x:i for i,x in enumerate(motifs)}
	motif_arr = np.zeros([len(tuples), len(motifs)])
	for i, tuple in enumerate(tuples):
		js = [m2i[x] for x in tuple]
		motif_arr[i,js] = 1

	# corr
	motif_corr = motif_arr.T @ motif_arr
	return motif_corr

def gen_df_arr_r():
	'''
	gen seqs with different lengths
	'''
	dff = df_arr[df_arr['level']=='global']
	df_arr_r = dff[['prog_id', 'motifs', 'seqs']]
	seqs_lists = [[np.unique([y[:l] for y in x]) for x in dff['seqs']] for l in range(1,6)]

	# df
	for i,l in enumerate(range(1,6)): df_arr_r['ngrams_%s'%l] = seqs_lists[i]

	# tuples as concat
	tuples = np.unique([tuple(sorted(x)) for x in fl(df_arr_r[['ngrams_1', 'ngrams_2', 'ngrams_3', 'ngrams_4', 'ngrams_5']].values)])
	motifs = sorted(np.unique(fl(tuples)), key=len)
	return df_arr_r, tuples, motifs


##%% RUN: original motif decomposition (12s)
motifs_MD = np.unique(df_motif['motif'].values)
seqs_MD = np.unique(df_seq['seq'].values)
tuples_MD = np.unique(df_seq['motifs'].values)
# tuples_MD = np.unique([tuple(sorted(x)) for x in df_arr['motifs']]).size # alternative measure
df_arr_r, tuples, motifs = gen_df_arr_r()

# motif_corr (30s)
motif_corr_MD = gen_motif_corr(motifs_MD, tuples_MD)
motif_corr = gen_motif_corr(motifs, tuples)

# sort by num motifs in common tuple
idx_sort_MD = (motif_corr_MD>0).sum(1).argsort()[::-1]

idx_len = np.cumsum([0] + [len([x for x in motifs if len(x)==l]) for l in range(1,6)])
idx_sort = (motif_corr>0).sum(1).argsort()[::-1]
idx_sort_0 = [(motif_corr>0).sum(1)[idx_len[i]:idx_len[i+1]].argsort()[::-1] for i in [0,1,2,3,4]]
idx_sort = fl([x+y for x,y in zip(idx_len[:-1], idx_sort_0)])

# attr
lengths = range(1,10)
num_complete = [4**l for l in lengths]
num_seqs = [np.unique([x[:l] for x in seqs_MD]).size for l in lengths]
num_motifs = [len([x for x in motifs_MD if len(x)==l]) for l in lengths]

colors = [cm.get_cmap('tab20b')(i) for i in [1,2,3]]

#%% plot
plt.figure(figsize=(18,3.5), dpi=300)
plt.subplot(131)
# plt.semilogy(lengths, num_motifs, label='# motifs', color='tab:cyan')
plt.semilogy(lengths, num_seqs, label='# sequences', color='k')
plt.semilogy(lengths, num_complete, label='# complete', color='k', linestyle='--')
plt.bar(lengths[:5], num_seqs[:5], color='tab:gray', width=.5)
plt.bar(np.array(lengths)+.2, num_motifs, color='tab:cyan', width=.5)
#
plt.scatter([9], num_seqs[-1], label='# tuples', s=70, color='k', marker='o', edgecolor='k')
plt.scatter([9], [len(tuples_MD)], label='# tuples', s=70, color='tab:cyan', marker='o', edgecolor='k')
plt.scatter([9], [len(tuples)], label='# tuples', s=70, color='tab:gray', marker='o', edgecolor='k')
#
plt.axvline(x=np.mean([len(x) for x in motifs_MD]), color='deeppink', linestyle='--', linewidth=1, label='avg motif len')
plt.xlabel('length')
plt.ylabel('number')
plt.legend()

plt.subplot(132)
plt.imshow(motif_corr_MD[idx_sort_MD,:][:,idx_sort_MD]>0, cmap='Greys', vmin=0)
plt.xlabel('motif j')
plt.ylabel('motif i')

plt.subplot(133)
plt.imshow(motif_corr[idx_sort,:][:,idx_sort]>0, cmap='Greys', vmin=0)
plt.xlabel('motif j')
plt.ylabel('motif i')

plt.savefig('fig/whyMotif.svg')


##%% PLOT: hist # motif per program
tuple_len_list = [[len(x) for x in tuples if len(x[0])==l] for l in range(1,6)]
num_motif_per_prog_max = max(fl(tuple_len_list))
hist_list = [[tuple_len.count(y) for y in range(1,num_motif_per_prog_max+1)] for tuple_len in tuple_len_list]
hist_MD = [[len(x) for x in df_arr_r['motifs']].count(y) for y in range(1,num_motif_per_prog_max+1)]

# plot
plt.figure(figsize=(3.5,3.5), dpi=300)
plt.subplot(6,1,1)
plt.bar(range(1,num_motif_per_prog_max+1), hist_MD, color='tab:cyan')
plt.xticks([])
plt.yticks([])

for i in range(5):
	plt.subplot(6,1,i+2)
	plt.bar(range(1,num_motif_per_prog_max+1), hist_list[i], color='tab:gray', label='# motif length = %s'%(i+1))
	if not i==4: plt.xticks([])
	else: plt.xlabel('# motif per program')
	plt.yticks([])
	# plt.legend()

plt.savefig('fig/whyMotif_num motifs per prog.svg')


##%% PLOT: hist avg motif len per program
motif_len_per_prog_list = [np.mean([len(y) for y in x]) for x in df_arr['motifs']]
len_uni = np.unique(motif_len_per_prog_list)
count_len = [motif_len_per_prog_list.count(x) for x in len_uni]

plt.figure(figsize=(4,3.5), dpi=300)
plt.hist(motif_len_per_prog_list, color='tab:cyan', bins=100)
plt.xlabel('mean motif len per program')
plt.ylabel('count')

plt.savefig('fig/whyMotif_mean motif len per prog.svg')


##%% PRINT: sorted motifs
for i,m in enumerate(motifs_MD[idx_sort_MD]): print('%s: %s'%(i,m))
print(len(motifs_MD), len(seqs_MD), len(tuples_MD))
len(motifs)

for i,m in enumerate(np.array(motifs)[idx_sort]): print('%s: %s'%(i,m))

##%% PRINT: motif enumeration (40s)
dff = df_seq[df_seq['level']=='global']
tuples_W = [x for x in dff['motifs'] if 'W' in x]
motifs_W = np.unique(fl(tuples_W))
count_W = [fl(tuples_W).count(x) for x in motifs_W]
idx_sort = np.argsort(count_W)[::-1]

mW_sort = motifs_W[idx_sort]
mW2rank = {x:i for i,x in enumerate(mW_sort)}

dff_arr = df_arr[(df_arr['level']=='global')]['motifs']
dfff_arr = dff_arr[[len(x)==3 for x in dff_arr]]
mutants = set([tuple(x) for x in dfff_arr[[{'W','l'}.issubset(set(x)) for x in dfff_arr]].values])

#
mutant_motif_list = np.unique(fl([[y for y in x if y not in ['W','l']] for x in mutants]))
bottom_rank = max([mW2rank[x] for x in mutant_motif_list])

#
dff_arr = df_arr[df_arr['level']=='global']
mutant_programs = fl([dff_arr[[tuple(y)==x for y in dff_arr['motifs']]]['prog_id'].values for x in mutants])
mut_prog_id_dict = {x:dff_arr[[tuple(y)==x for y in dff_arr['motifs']]]['prog_id'].values for x in mutants}
num_mut_prog_dict = {x:dff_arr[[tuple(y)==x for y in dff_arr['motifs']]]['prog_id'].values.size for x in mutants}

print(mutants)
print(mutant_programs, len(mutant_programs))
print(len(mutant_motif_list), bottom_rank)
print(mW_sort[2:48])

##%% PRINT: more motif stat
tuples = [tuple(sorted(x)) for x in df_arr[(df_arr['level']=='global')]['motifs']]
len([x for x in tuples if x==('W','l')])
sorted(tuples, key=len)

np.unique(df_seq[df_seq['seq']=='LlllWlLll']['motifs'])
np.unique(df_seq[df_seq['seq']=='LlllWlLll']['prog_id']).size
