'''
for DB_size_max = 8
p_thres=.5: 40s
p_thres=1: 12m on laptop
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
from core_enum import gen_m_groups_dict
from core_enumDB import gen_enumDB_list_with_discretization_sweep

## parameters
para_id = 14
DB_size_max = 8
p_thres = [.5, 1-1e-12, 1][-1]


## load
df_task = pd.read_pickle('data/df_task')
para = df_task.loc[para_id, ['h','dp','dpm']].values
try:
	m_groups_dict = pickle.load(open('data/m_groups_dict_enumDB','rb'))
except FileNotFoundError:
	print('generating m_groups_dict_enumDB... from merger rule')
	m_groups_dict = gen_m_groups_dict(prog_size_max=8, for_enumDB=True)
	pickle.dump(m_groups_dict, open('data/m_groups_dict_enumDB', 'wb'))


## RUN
if False:
	enumDB_list = gen_enumDB_list_with_discretization_sweep(para, m_groups_dict, DB_size_max=DB_size_max, p_thres=p_thres)
	print('total =', len(enumDB_list))

	# df
	progsize_list = [len(x[0]) for x in enumDB_list]
	df = pd.DataFrame([enumDB_list, progsize_list]).T
	df.columns = ['program', 'progsize']
	pd.to_pickle(df, 'data/df_enumDB_para14')


# ## PRINT
# df = pd.read_pickle('data/df_enumDB_para14')
# len(df)
