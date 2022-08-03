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
if not os.path.exists('fig'): os.makedirs('fig')


## LOAD
df_pte = pd.read_pickle('data/df_gpn_pte')
tar_list = df_pte['target'].values
pprog_arr = pickle.load(open('data/df_gpn_pprog_arr', 'rb'))


##%% PLOT
tars_iden = df_pte[df_pte['is_iden']==1].index.values
ids_iden = [tar_list.tolist().index(x) for x in tars_iden]
pprog_arr_r = pprog_arr[ids_iden,:,-1][:,ids_iden]
cl_dict = {i:x for i,x in enumerate(df_pte.loc[tars_iden, 'fn_comm'].values)}

num_cl = max(list(cl_dict.values())) + 1
blocks = [[] for i in range(num_cl)]
for id_prog, id_blk in cl_dict.items(): blocks[id_blk].append(id_prog)
id_blk = fl(blocks)
#
plt.figure(figsize=(10,4), dpi=300)
plt.title('pprog, num_cl=%s'%num_cl)
plt.imshow(pprog_arr_r[id_blk,:][:,id_blk], cmap='Greys', alpha=1, vmax=pprog_arr_r.mean()+pprog_arr_r.std())
plt.xlabel('sorted prog j')
plt.ylabel('sorted prog i')
plt.colorbar()
plt.xticks([])
plt.yticks([])
plt.savefig('fig/fn_comm_pprog.svg')
