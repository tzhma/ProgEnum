'''
...
performance_min, performance_max = 1.000, 0.069
'''
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


## FUNCTION
def gen_venn(job_id):
    # get seq_B
    seq_B = dfB.loc[job_id-1, 'seq']

    # get seq_enumTaskP
    df = pd.read_pickle('data/df_enumTaskP_r/df_enumTaskP_%s'%job_id)
    eRWSLG = df[df['progsize']==2]['eR'].max()
    dff = df[df['eR']>=eRWSLG]
    seq_enumTaskP = np.unique(fl(df14.loc[dff.index, 'seq'].values))

    # venn WSLG
    venn_WSLG = len(set(seq_B).difference(set(seq_WSLG))), len(set(seq_B).intersection(set(seq_WSLG))), len(set(seq_WSLG).difference(set(seq_B)))

    # venn df_enumTaskP
    venn_enumTaskP = len(set(seq_B).difference(set(seq_enumTaskP))), len(set(seq_B).intersection(set(seq_enumTaskP))), len(set(seq_enumTaskP).difference(set(seq_B)))
    return venn_WSLG, venn_enumTaskP


## MP (15s)
job_id_list = range(1,69)

venn_WSLG_, venn_enumTaskP_ = [], []
with multiprocess.Pool(num_cpus) as p:
    jobs = p.map(gen_venn, job_id_list)
    for k, job in enumerate(jobs):
        venn_WSLG_.append(job[0])
        venn_enumTaskP_.append(job[1])
        print(k+1, job)

# df
df = pd.DataFrame([h_list, venn_WSLG_, venn_enumTaskP_]).T
df.columns = ['h', 'venn_WSLG', 'venn_enumTaskP']


##%% PLOT
nnz_B_excl_WSLG, nnz_intersec_WSLG, nnz_excl_WSLG = np.array(df['venn_WSLG'].values.tolist()).T
nnz_B_excl_enumTaskP, nnz_intersec_enumTaskP, nnz_excl_enumTaskP = np.array(df['venn_enumTaskP'].values.tolist()).T

nnz_B = nnz_B_excl_WSLG + nnz_intersec_WSLG
nnz_WSLG = nnz_excl_WSLG + nnz_intersec_WSLG
nnz_enumTaskP = nnz_excl_enumTaskP + nnz_intersec_enumTaskP

plt.figure(figsize=(10,8), dpi=300)
plt.subplot(211)
plt.semilogy(h_list, nnz_B, label='nnz_B', color='tab:cyan')
plt.semilogy(h_list, nnz_WSLG, label='nnz', color='k')
plt.semilogy(h_list, nnz_intersec_WSLG, label='nnz_intersec', color='#008B8B')
plt.axvline(x=h_c0, color='deeppink', linestyle='--', linewidth=1)
plt.axvline(x=h_c1, color='deeppink', linestyle='--', linewidth=1)
plt.axvline(x=h_c2, color='deeppink', linestyle='--', linewidth=1)
plt.axvline(x=h_c3, color='deeppink', linestyle='--', linewidth=1)
plt.xlabel('h')
plt.ylabel('# seq')
plt.legend()

plt.subplot(212)
plt.semilogy(h_list, nnz_B_excl_enumTaskP+nnz_intersec_enumTaskP, label='nnz_B', color='tab:cyan')
plt.semilogy(h_list, nnz_excl_enumTaskP+nnz_intersec_enumTaskP, label='nnz', color='k')
plt.semilogy(h_list, nnz_intersec_enumTaskP, label='nnz_intersec', color='#008B8B')
plt.axvline(x=h_c0, color='deeppink', linestyle='--', linewidth=1)
plt.axvline(x=h_c1, color='deeppink', linestyle='--', linewidth=1)
plt.axvline(x=h_c2, color='deeppink', linestyle='--', linewidth=1)
plt.axvline(x=h_c3, color='deeppink', linestyle='--', linewidth=1)
plt.xlabel('h')
plt.ylabel('# seq')
plt.legend()

plt.savefig('fig/taskSloppiness_enumTaskP_seqVenn.svg')

#%% PRINT (2bsorted)
dfB[dfB['h']==.358]['pB_WLl'].values

[x for x in dfB[dfB['h']==.358]['seq'].values[0] if 'L' in x][:5]
[x for x in dfB[dfB['h']==.358]['seq'].values[0] if 'L' in x][-5:]

h_id = h_list.index(.334)
df = pd.read_pickle('data/df_enumTaskP_r/df_enumTaskP_%s'%h_id)
eRWSLG = df[df['progsize']==2]['eR'].max()
dff = df[df['eR']>=eRWSLG]
seq_enumTaskP = np.unique(fl(df14.loc[dff.index, 'seq'].values))

len([x for x in seq_enumTaskP if 'w' in x])
nnz_excl_enumTaskP[34] - len([x for x in seq_enumTaskP if 'w' in x])
[x for x in seq_enumTaskP if 'w' in x][:5]
[x for x in seq_enumTaskP if 'w' in x][-5:]
