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
from core_enum import fl
from core_bdp import run_bdp
from core_sdp import run_sdp, gen_pprog_lists, gen_denom_list


## LOAD: prog_ids of local cluster 9
df_pte = pd.read_pickle('data/df_gpn_pte')
tar_list = df_pte['target'].values
tars_9 = [9] + df_pte[df_pte['source']==9].index.values.tolist()
ids_9 = [tar_list.tolist().index(x) for x in tars_9]

# extract local seq-hist
Y_lists = pickle.load(open('data/df_gpn_Y_lists', 'rb'))
Y0_list = Y_lists[ids_9[0]]
Y1_lists = [Y_lists[x] for x in ids_9]

# compute local pp
denom_list = [np.stack(Y_lists[tar_list.tolist().index(tar)][tau].todense() for tar in tars_9).sum(0)+1e-16 for tau in range(10)]
pprog_list = gen_pprog_lists(Y1_lists, Y0_list, denom_list)
pprog_arr = np.stack(pprog_list)[None,:,:]


## FUNCTIONS
def gen_simprog_arr():
	'''
	1m
	'''
	pp1 = np.zeros_like(pprog_arr)
	for t in range(pprog_arr.shape[-1]):
		for i in range(len(pprog_arr)):
			pp1[i, np.argsort(pprog_arr[i,:,t])[::-1][:int(np.around(1/pprog_arr[i,i,t]))], t] = 1
	#
	pp2 = pp1#((pp1 + np.transpose(pp1, [1,0,2]))>0)*1
	pp3 = pp2.sum(-1)
	return pp2, pp3

## run
pp2, pp3 = gen_simprog_arr()
# pickle.dump(pp2, open('data/df_gpn_simprog_arr_list_local_P9', 'wb'))
# pickle.dump(pp3, open('data/df_gpn_simprog_arr_local_P9', 'wb'))

## df
df = pd.DataFrame([np.array(tars_9).astype('int'), pp3[0].astype('int')]).T
df.columns = ['prog_id', 'pp_sim']
df.to_pickle('data/df_gpn_local_keymut')
