'''
NOTE: this code extracts a minimal single-component subtree covering all good programs (better than wslg,  including wslg)
INPUT: pte (& its rpte) that excludes lowest three P2s
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
import networkx as nx
from collections import ChainMap
import multiprocess
from core_enum import fl, reshape_prog, gen_stdz_prog
from core_pte import gen_perm_prog_array, gen_edit_distance_list, gen_one_merger_prog_list

num_cpus = int(multiprocess.cpu_count()*1)
print('num_cpus = %s'%num_cpus)


## LOAD
extension = '268533'
df = pd.read_pickle('data/df_rpte_%s_branch'%extension)
df.index = df['target'].values


## RUN
eR_thres = 0.277911
tar_list = df[df['eR']>=eR_thres]['target'].values
tar_all_list = sorted(set(fl(df.loc[tar_list, 'branch']))) # 4492
tar_conn_list = [x for x in tar_all_list if x not in tar_list] # 262 (5.8%)

## df
df_gpn = df.loc[tar_all_list]
df_gpn.to_pickle('data/df_gpn')
