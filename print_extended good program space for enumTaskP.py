'''
Q: can .99*eR_WSLG encompass all good programs in higher h?
A: should be! for only handful of non-para14-good programs found (3 out of 2731 at most; lower than eR_WSLG by factor of .9939)

2731 programs > WSLG at para14

PRINT: can .99*eR_WSLG encompass all good programs in higher h? --> should be! for only handful of non-para14-good programs found (3 out of 2731 at most; lower than eR_WSLG by factor of .9939)
'''
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import os
import sys
from itertools import permutations, combinations, product
import pandas as pd
from core_bdpB import run_bdpB


## PARAMETERS
dp, dpm = .3, -.5
h_list = np.around(np.linspace(.05, .32, 28), 6)[:-1].tolist() + np.around(np.linspace(.32, .4, 41), 6).tolist()


## LOAD
df14 = pd.read_pickle('data/df_enumP_para14')
eR14 = df14[df14['progsize']==2]['eR'].max()
uni14 = df14[df14['eR']>=eR14].index.values


## PRINT
num_prog_list = []
eR_max_list = []
for k, h in enumerate(h_list):
	job_id = k+1
	if job_id==1: df = pd.read_pickle('data/df_enumP_para14')
	else: df = pd.read_pickle('data/df_enumTaskP_r/df_enumTaskP_%s'%job_id)
	eR0 = df[df['progsize']==2]['eR'].max()
	uni = df[df['eR']>=eR0].index.values
	eRs = [np.around(df14[df14.index==x]['eR'].values[0]/eR14,4) for x in uni if x not in uni14]
	print(h, eRs)
