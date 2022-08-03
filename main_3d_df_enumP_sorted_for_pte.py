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
import plotly.express as px


## functions
def fl(l): return [x for y in l for x in y]

def gen_df_sorted(df, filename, sort_type=1):
	'''
	generate different versions of sorted df_enumP_para14
	1) df.sort_values(['progsize','eR'], ascending=[True, False])
	2) df.sort_values(['progsize','eR'], ascending=[True, half-randomized])
	3) df.sort_values(['progsize','eR'], ascending=[True, randomized])
	'''
	if sort_type==1:
		df['id_unique'] = df.index.values # id_unique are fixed across different sorting
		dff = df.sort_values(['progsize', 'eR'], ascending=[True, False])
		dff.index = range(len(dff))

	if sort_type==2:
		df['id_unique'] = df.index.values # id_unique are fixed across different sorting
		dff = df.sort_values(['progsize','eR'], ascending=[True, False])
		dff.index = range(len(dff))

		# randomizing eR above and below eR_rand
		np.random.seed(42)
		idx_group_list = []
		for progsize in [2,3,4,5]:
			idx_group_good = sorted(dff[(dff['progsize']==progsize) & (dff['eR']>=.25)].index.values)
			idx_group_bad = sorted(dff[(dff['progsize']==progsize) & (dff['eR']<.25)].index.values)
			np.random.shuffle(idx_group_good)
			np.random.shuffle(idx_group_bad)
			idx_group_list.append(idx_group_good)
			idx_group_list.append(idx_group_bad)

		# reindex
		dff = dff.loc[fl(idx_group_list)]
		dff.index = range(len(dff))

	if sort_type==3:
		df['id_unique'] = df.index.values # id_unique are fixed across different sorting
		idx_rand = sorted(df.index.values)
		np.random.seed(42)
		np.random.shuffle(idx_rand)
		dff = df.loc[idx_rand].sort_values('progsize', ascending=True)
		dff.index = range(len(dff))
	return dff


## load
df_task = pd.read_pickle('data/df_task')
df_enumP = pd.read_pickle('data/df_enumP_para14')

if False:
	df_1 = gen_df_sorted(df_enumP, 'df_enumP_para14', sort_type=1)
	df_2 = gen_df_sorted(df_enumP, 'df_enumP_para14', sort_type=2)
	df_3 = gen_df_sorted(df_enumP, 'df_enumP_para14', sort_type=3)

	# pickle
	filename = 'df_enumP_para14'
	pd.to_pickle(df_1, 'data/%s_sorted'%filename)
	pd.to_pickle(df_2, 'data/%s_halfrand'%filename)
	pd.to_pickle(df_3, 'data/%s_rand'%filename)
