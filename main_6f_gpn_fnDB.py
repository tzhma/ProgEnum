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
from core_bdp import run_bdp
from core_sdp import run_sdp, gen_pprog_lists, gen_denom_list


## LOAD
df_pte = pd.read_pickle('data/df_gpn_pte')
tar_list = df_pte['target'].values
Y_lists = pickle.load(open('data/df_gpn_Y_lists','rb')) # 10 ao_hists for all programs


## FUNCTIONS: gen_Y_megaDB
def gen_Y_megaDB():
	prog_ids_DB = df_pte[df_pte['d2enumDB']==0].index.values
	Y_DBs = [Y_lists[x][-1] for x in [tar_list.tolist().index(y) for y in prog_ids_DB]]
	Y_megaDB = np.stack(Y_DBs).sum(0)
	# Y_megaDB = Y_megaDB/Y_megaDB.sum()
	# pickle.dump(Y_megaDB, open('data/Y_megaDB', 'wb'))
	return Y_megaDB

def gen_pp_DB_list(Y_megaDB):
	'''
	compute local pp 46s
	'''
	pp_list = []
	for x in range(len(tar_list)):
		Y = Y_lists[x][-1].todense()
		pp = (Y**2/(Y + Y_megaDB + 1e-16)).sum()
		pp_list.append(pp)
	return pp_list


##%% RUN
if False:
	Y_megaDB = gen_Y_megaDB()
	pp_list = gen_pp_DB_list(Y_megaDB)

	# df
	prog_ids_DB = df_pte[df_pte['d2enumDB']==0].index.values
	ids_DB = [tar_list.tolist().index(x) for x in prog_ids_DB]
	pp_DB_thres = np.sort(np.array(pp_list)[ids_DB])[-3] # excluding two outlier
	df_pte['pp_DB'] = pp_list
	df_pte['is_fn_DB'] = np.array(pp_list)<=pp_DB_thres

	# pickle
	pd.to_pickle(df_pte, 'data/df_gpn_pte')
