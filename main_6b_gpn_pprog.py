'''
NOTE:
1) run mp on cluster (45 jobs): 10m
2) assemble them into a single file: df_gpn_pprog_arr
'''
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex, ListedColormap, LinearSegmentedColormap
from matplotlib import cm
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
import multiprocessing
import sparse


num_cpus = int(multiprocessing.cpu_count()*1)
print('num_cpus = %s'%num_cpus)
if not os.path.exists('data'): os.makedirs('data')


job_id = int(sys.argv[1]) # 1~45 for 100 Y_list each


## LOAD
Y_lists = pickle.load(open('data/df_gpn_Y_lists', 'rb'))
Y0_lists = Y_lists[(job_id-1)*100: job_id*100]
try:
	denom_list = pickle.load(open('data/denom_list', 'rb'))
except FileNotFoundError:
	denom_list = gen_denom_list(Y_lists)
	pickle.dump(denom_list, open('data/denom_list', 'wb'))


## MP
if False:
	def run_mp3(arg):
		Y_lists, Y0_list, denom_list = arg
		pprog_lists = gen_pprog_lists(Y_lists, Y0_list, denom_list)
		return pprog_lists

	pprog_arr = []
	with multiprocessing.Pool(num_cpus) as p:
		jobs = p.map(run_mp3, [(Y_lists,x,denom_list) for x in Y0_lists])

		for i, job in enumerate(jobs):
			pprog_arr.append(job)

			if i % max(len(Y_lists)//100, 1)==0:
				print(i, len(job[-1]))

	# pickle
	pickle.dump(pprog_arr, open('data/df_gpn_pprog_arr_%s'%job_id, 'wb'))


## ASSEMBLE
if False:
	pprog_arr = []
	for job_id in range(1,46):
		pprog = pickle.load(open('data/df_gpn_pprog_arr_%s'%job_id, 'rb'))
		pprog_arr += pprog
	pprog_arr = np.array(pprog_arr)
	pprog_arr.shape

	# pickle
	pickle.dump(pprog_arr, open('data/df_gpn_pprog_arr', 'wb'))
