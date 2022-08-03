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
from core_bdp import run_bdp
from core_sdp import run_sdp, gen_pprog_lists, gen_denom_list
from core_enum import fl
import community as community_louvain

## PARAMETERS
rho = 1 # resolution for community_louvain
np.random.seed(42)


## LOAD
df_pte = pd.read_pickle('data/df_gpn_pte')
pprog_arr = pickle.load(open('data/df_gpn_pprog_arr', 'rb'))
tar_list = df_pte['target'].values


##%% RUN: cluster iden programs
if False:
	tars_iden = df_pte[df_pte['is_iden']==1].index.values
	ids_iden = [tar_list.tolist().index(x) for x in tars_iden]
	pprog_arr_r = pprog_arr[ids_iden,:,-1][:,ids_iden]

	# community_louvain
	G = nx.from_numpy_array(pprog_arr_r)
	cl_dict = community_louvain.best_partition(G, weight='weight', resolution=rho)

	# df
	tar_cl_dict = {tars_iden[id]:cl for id, cl in cl_dict.items()}
	cl_list = []
	for tar in tar_list:
		try: cl = tar_cl_dict[tar]
		except KeyError: cl = -1
		cl_list.append(cl)
	df_pte['fn_comm'] = np.array(cl_list) - df_pte['is_fn_DB']*1


	# pickle
	col4csv = ['id', 'id_unique', 'target', 'source', 'progsize', 'eR', 'd2enumDB', 'iden', 'd2root', 'num_mut', 'ecc', 'pp_sim', 'pp_DB', 'weight', 'is_iden', 'is_fn_DB', 'fn_comm']
	df_pte[col4csv].to_csv('gephi/df_gpn_pte.csv', index=False)
	df_pte.to_pickle('data/df_gpn_pte')
