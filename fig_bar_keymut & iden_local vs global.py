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


##%% PLOT: local vs global pp
pp_global = np.array([df_global[df_global['prog_id']==x]['pp'].sum() for x in idx9])
pp_local = np.array([df_local[df_local['prog_id']==x]['pp'].sum() for x in idx9])
pp_local[0] = df_seq[(df_seq['prog_id']==9) & (df_seq['level']=='parent')]['pp'].sum()
is_keymut = ((df_key['pp_sim']<10).values*50).astype('float')

filter_24 = np.zeros_like(pp_local)
filter_24[2] = 1
filter_fnDB = np.zeros_like(pp_local)
filter_fnDB[ids_fnDB] = 1
#
plt.figure(figsize=(12,6), dpi=300)
plt.subplot(211)
plt.bar(range(len(pp_local)), pp_local, color='#e6e6e6', zorder=0)
plt.bar(range(len(pp_local)), pp_local*filter_fnDB, color='#B3B3B3', zorder=0)
plt.bar(range(len(pp_local)), pp_local*filter_24, color='tab:purple', zorder=0)
#
plt.scatter(range(len(pp_local)), pp_local+.002, marker="*", s=is_keymut, label='keymut', color='k', edgecolor='none', zorder=1)
plt.axhline(y=pp_local.mean(), color='r', linestyle='--', label='mean(pp)', zorder=0)
plt.xticks(range(len(pp_local)), labels=idx9, rotation='-45')
plt.ylabel('local pp')
plt.legend()
#
plt.subplot(212)
plt.bar(range(len(pp_global)), pp_global, color='#e6e6e6', zorder=0)
plt.bar(range(len(pp_global)), pp_global*filter_fnDB, color='#B3B3B3', zorder=0)
plt.bar(range(len(pp_global)), pp_global*filter_24, color='tab:purple', zorder=0)
plt.xticks(range(len(pp_global)), labels=idx9, rotation='-45')
plt.ylabel('global pp')
plt.tight_layout()
plt.savefig('fig/keymut_local_global_pp.svg')


##%% PLOT: pp with top pp-seq from 24, 6592, 8261
ids_top3 = [idx9.tolist().index(x) for x in [24, 6592, 8261]]
dpp_top3 = pp_local[ids_top3] - pp_local.mean()

num50_top3 = [np.where((df_local[df_local['prog_id']==x]['pp'].cumsum() - y)>0)[0][0] for x,y in zip([24, 6592, 8261], dpp_top3)]
seqs_top3 = [df_local[df_local['prog_id']==x]['seq'][:y].values for x,y in zip([24, 6592, 8261], num50_top3)]
seqs_motifs_top3 = [df_local[df_local['prog_id']==x][['seq','motifs']][:y].sort_values('motifs').values for x,y in zip([24, 6592, 8261], num50_top3)]
motifs_top3 = [np.unique([y[1] for y in x]) for x in seqs_motifs_top3]

#
seqs_motifs_top3_dicts = [{} for x in seqs_top3]
for k, seqs_motifs in enumerate(seqs_motifs_top3):
	for seq, motif in seqs_motifs:
		try: seqs_motifs_top3_dicts[k][motif] += [seq]
		except KeyError: seqs_motifs_top3_dicts[k][motif] = [seq]

pp_local_top3 = [np.array([df_local[[x in y for x in df_local['seq']] & (df_local['prog_id']==x)]['pp'].sum() for x in idx9]) for y in seqs_top3]

#
plt.figure(figsize=(12,4), dpi=300)
for i, pp_top in enumerate(pp_local_top3):
	plt.subplot(3,1,i+1)
	plt.bar(range(len(pp_local)), pp_top, color='#e6e6e6', zorder=0)
	plt.bar(range(len(pp_local)), pp_top*filter_fnDB, color='#B3B3B3', zorder=0)
	plt.bar(range(len(pp_local)), pp_top*filter_24, color='tab:purple', zorder=0)
	#
	plt.xticks([])
	plt.ylabel('local pp')
plt.tight_layout()
plt.savefig('fig/keymut_local_pp_with_top_seqs.svg')


##%% PRINT: top seqs
for x in seqs_motifs_top3_dicts:
	print('')
	print(x)


##%% PRINT: 6592 (parent: 9) vs 5982
seqs = set(df_seq[(df_seq['prog_id']==5982) & (df_seq['level']=='global')].sort_values('pp', ascending=False)['seq']).intersection(set(df_seq[(df_seq['prog_id']==6592) & (df_seq['level']=='global')].sort_values('pp', ascending=False)['seq']))

print(seqs)

for seq in seqs:
	if seq not in seqs_top3[1]: continue

	_df = df_seq[[x in [5982, 6592] for x in df_seq['prog_id']] & (df_seq['level']=='global') & [x in seq for x in df_seq['seq']]].sort_values('pp', ascending=False)[['seq','motifs','pp','prog_id']]
	print(_df)


##%% PRINT: 8261 (parent: 9) vs 6118
seqs = set(df_seq[(df_seq['prog_id']==6118) & (df_seq['level']=='global')].sort_values('pp', ascending=False)['seq']).intersection(set(df_seq[(df_seq['prog_id']==8261) & (df_seq['level']=='global')].sort_values('pp', ascending=False)['seq']))

print(seqs)

for seq in seqs:
	if seq not in seqs_top3[2]: continue

	_df = df_seq[[x in [6118, 8261] for x in df_seq['prog_id']] & (df_seq['level']=='global') & [x in seq for x in df_seq['seq']]].sort_values('pp', ascending=False)[['seq','motifs','pp','prog_id']]
	print(_df)
