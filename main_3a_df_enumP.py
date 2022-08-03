'''
single: 43m
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
from core_enum import gen_outmap_list, gen_m_groups_dict, gen_inmap_fl_list_with_filters, gen_program_list_from_inmap_fl_list


## LOAD
try:
	m_groups_dict = pickle.load(open('data/m_groups_dict_enumP','rb'))
except FileNotFoundError:
	m_groups_dict = gen_m_groups_dict(prog_size_max=8)
	pickle.dump(m_groups_dict, open('data/m_groups_dict_enumP', 'wb'))


## RUN
if False:
	outmap_list = gen_outmap_list(prog_size_max=5)
	program_list = []
	for outmap in outmap_list:
		inmap_fl_list = gen_inmap_fl_list_with_filters(outmap, m_groups_dict)
		program_list += gen_program_list_from_inmap_fl_list(inmap_fl_list, outmap)

	# df
	progsize_list = [len(x[0]) for x in program_list]
	df = pd.DataFrame([program_list, progsize_list]).T
	df.columns = ['program', 'progsize']
	pd.to_pickle(df, 'data/df_enumP')
