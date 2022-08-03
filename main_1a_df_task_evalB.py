'''
1) run_bdpB to find optimal performance of full-B
2) append three new columns to df_param: eR_B, t_iter_B, p_array_B
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
from core_bdpB import run_bdpB


## function
def gen_df_eR_B(df):
	'''
	input: data/df_task (run main_0_df_task first)
	find eR for full-B
	'''
	# load
	if use_param_fine: df = pd.read_pickle('data/df_task_fine')
	else: df = pd.read_pickle('data/df_task')

	try:
		df['eR_B']
		print('eR_B data already exist')
		return df

	except KeyError:
		eR_B_list, t_iter_B_list, p_array_B_list = [], [], []
		for i in df.index:
			para = df.loc[i, ['h','dp','dpm']]
			eR, p_array, t_iter= run_bdpB(para)
			eR_B_list.append(eR)
			t_iter_B_list.append(t_iter)
			p_array_B_list.append(p_array)
			if i % max(len(df)//10, 1)==0: print('param', i, eR, t_iter)

		# df
		df['eR_B'] = eR_B_list
		df['t_iter_B'] = t_iter_B_list
		df['p_array_B'] = p_array_B_list
	return df


##%% RUN
if False:
	df = gen_df_eR_B()
	pd.to_pickle(df, 'data/df_task')


## PLOT
df = pd.read_pickle('data/df_task')

dff = df[df['h']==.05]
plt.figure()
plt.scatter(dff['dpm'], dff['dp'], c=dff['eR_B'])
plt.colorbar()

dff = df[np.abs(df['h']-.05)<1e-6]
plt.figure()
plt.scatter(dff['dpm'], dff['dp'], c=dff['t_iter_B'])
plt.colorbar()
