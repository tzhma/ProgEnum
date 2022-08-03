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


## LOAD: prog_ids of local cluster 9
df_pte = pd.read_pickle('data/df_gpn_pte')
df_seq = pd.read_pickle('data/df_gpn_seq_motif_1')
df_motif = pd.read_pickle('data/df_gpn_seq_motif_2')
df_key = pd.read_pickle('data/df_gpn_local_keymut')


# get data (40s)
idx9 = df_key['prog_id'].values
df_local = df_seq[[x in idx9 for x in df_seq['prog_id']] & (df_seq['level']=='local')]
df_global = df_seq[[x in idx9 for x in df_seq['prog_id']] & (df_seq['level']=='global')]
ids_fnDB = np.array([idx9.tolist().index(x) for x in df_pte.loc[idx9][df_pte.loc[idx9]['is_fn_DB']].index.values])

# descendents of 24
idx24 = [24] + df_pte[df_pte['source']==24].index.values.tolist()
idx24_iden = [24, 25, 277, 400, 501, 815, 7655]
df24_global = df_seq[[x in idx24_iden for x in df_seq['prog_id']] & (df_seq['level']=='global')]
seqs24 = np.unique(df24_global['seq'])


## PLOT: motif-dist 9 vs 24
idx9_without24 = np.array(idx9[:2].tolist() + idx9[3:].tolist())
df9_motif = df_motif[[x in idx9_without24 for x in df_motif['prog_id']] & (df_motif['level']=='global')]
df24_motif = df_motif[[x in idx24_iden for x in df_motif['prog_id']] & (df_motif['level']=='global')]

motifs_9 = df9_motif['motif'].values.tolist()
motifs_24 = df24_motif['motif'].values.tolist()

motifs_all = np.unique(motifs_9 + motifs_24)

count_9 = np.array([motifs_9.count(x) for x in motifs_all])
count_24 = np.array([motifs_24.count(x) for x in motifs_all])

idx_sort = np.argsort(count_9 - count_24/100)[::-1]

# plot
plt.figure(figsize=(10,6), dpi=300)
plt.subplot(211)
plt.bar(range(len(motifs_all)), count_9[idx_sort], color='#B3B3B3', label='descendants of prog 9 (excluding prog 24)')
plt.xticks(range(len(motifs_all)), labels=motifs_all[idx_sort], rotation='-45')
plt.ylabel('# programs')
plt.legend()

plt.subplot(212)
plt.bar(range(len(motifs_all)), count_24[idx_sort], color='tab:purple', label='descendants of prog 24 (including prog 24)')
plt.xticks(range(len(motifs_all)), labels=motifs_all[idx_sort], rotation='-45')
plt.ylabel('# programs')
plt.legend()

plt.tight_layout()
plt.savefig('fig/motif_dist_9_vs_24.svg')


#%% PRINT: motifs from 24 & 7655
df_pte.loc[idx24, ['target', 'eR', 'is_iden']]
df_motif[(df_motif['level']=='global') & (df_motif['prog_id']==24)]['motif']
df_motif[(df_motif['level']=='global') & (df_motif['prog_id']==7655)]['motif']

set(df_seq[(df_seq['level']=='global') & (df_seq['prog_id']==24)]['motifs'])
set(df_seq[(df_seq['level']=='global') & (df_seq['prog_id']==7655)]['motifs'])

df_seq[(df_seq['level']=='global') & (df_seq['prog_id']==24)][['seq', 'motifs']][:50]


##%% PRINT
for seq in seqs24:
	_df = df24_global[df24_global['seq']==seq][['seq', 'motifs', 'prog_id']]
	if len(_df)<3: continue
	print(_df)
	print('')
