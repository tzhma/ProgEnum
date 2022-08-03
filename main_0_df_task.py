'''
This generate df of task parameters
1) coarse-grained sweep: 330 parameters
2) fine-grained sweep: 1155 parameters
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


## function
def gen_df_task(fine_grain=False):
	if fine_grain:
		p0_list = np.around(np.arange(0,1.0001,.05), 2)
		p1_list = np.around(np.arange(0,1.0001,.05), 2)
		h_list = np.around(np.arange(.05,.5,.1), 2)
	else:
		p0_list = np.around(np.arange(0,1.0001,.1), 2)
		p1_list = np.around(np.arange(0,1.0001,.1), 2)
		h_list = np.around(np.arange(.05,.5,.1), 2)
	# pd
	query = list(product(h_list, p0_list, p1_list))
	df = pd.DataFrame(query, columns=[ 'h','p0','p1'])
	dff = df[df['p0'] <= df['p1']]
	dff['dpm'] = np.around(dff['p0'] + dff['p1'] - 1, 6)
	dff['dp'] = np.around(dff['p1'] - dff['p0'], 6)
	dff.index = np.arange(len(dff))
	return dff


## run
if False:
	df = gen_df_task(fine_grain=False)
	pd.to_pickle(df, 'data/df_task')

	# df_fine = gen_df_task(fine_grain=True)
	# pd.to_pickle(df_fine, 'data/df_task_fine')


# ##%% PLOT
# import plotly.express as px
# import pandas as pd
#
# df = pd.read_pickle('data/df_task')
# fig = px.scatter_3d(df, x='dp', y='dpm', z='h', size_max=10, opacity=1, height=800, labels={'dp': '','dpm': '','h': ''})
# fig
# # fig.write_image("task_param.pdf")
