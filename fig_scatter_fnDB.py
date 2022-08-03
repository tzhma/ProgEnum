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

if not os.path.exists('fig'): os.makedirs('fig')


## LOAD
df_pte = pd.read_pickle('data/df_gpn_pte')
tar_list = df_pte['target'].values


##%% PLOT: idenDB vs idenGlobal
pp_list = df_pte['pp_DB'].values
prog_ids_DB = df_pte[df_pte['d2enumDB']==0].index.values
ids_DB = [tar_list.tolist().index(x) for x in prog_ids_DB]
pp_DB_thres = np.sort(np.array(pp_list)[ids_DB])[-3]
#
prog_ids_good = df_pte[df_pte['eR']>=.25].index.values
ids_good = [tar_list.tolist().index(x) for x in prog_ids_good]
prog_ids_bad = df_pte[df_pte['eR']<.25].index.values
ids_bad = [tar_list.tolist().index(x) for x in prog_ids_bad]
#
c_dict = {-2:'#B3B3B3', -1:'#e6e6e6', 0:'tab:blue', 1:'tab:orange', 2:'tab:green', 3:'tab:red', 4:'tab:purple', 5:'tab:brown', 6:'tab:pink', 7:'tab:olive', 8:'tab:cyan'}
c_list = np.array([c_dict[x] for x in df_pte['fn_comm'].values])
s_list = df_pte['iden'].values*0 + 10
#
plt.figure(figsize=(8,5), dpi=300)
iden_thres = df_pte[df_pte['is_iden']==1]['iden'].min()
iden_global = df_pte['iden'].values
plt.scatter(iden_global[ids_bad], pp_list[ids_bad], s=20, c='w', label='low-performing connection programs', edgecolor='dimgray', linewidth=.75)
plt.scatter(iden_global[ids_good], pp_list[ids_good], s=10, c=c_list[ids_good], label='GPN')
plt.scatter(iden_global[ids_DB], pp_list[ids_DB], s=10, c='k', label='DB')
plt.axvline(x=iden_thres, color='r', linestyle='--', label='iden to cover 50% pp')
plt.axhline(y=pp_DB_thres, color='k', linestyle='--', label='pp_max of nearly all DBs')
plt.xlabel('iden')
plt.ylabel('pp in pool of 65 DBs')
plt.grid()
plt.legend()
plt.savefig('fig/fnDB_pp_iden.svg')
