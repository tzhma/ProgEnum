'''
NOTE:
1) run mp on cluster (45 jobs): 5m
2) assemble them into a single file: dprog_part_arr, dprog_arr
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
import networkx as nx
import pandas as pd
from core_enum import fl
from core_bdp import run_bdp
from core_sdp import run_sdp, gen_pprog_lists, gen_eR_from_Y_list
import multiprocessing
from core_pte import gen_perm_prog_array, gen_edit_distance, gen_part_list_for_merger_prog

num_cpus = int(multiprocessing.cpu_count()*1)
print('num_cpus = %s'%num_cpus)


## LOAD
mergP_dict = pickle.load(open('data/mergP_dict','rb'))


# load GPN
df_gpn = pd.read_pickle('data/df_gpn')
tar_list = df_gpn['target'].values
sou_list = df_gpn['source'].values
id_uni_list = df_gpn['id_unique'].values
pprog_arr = pickle.load(open('data/df_gpn_pprog_arr', 'rb'))[:,:,-1]
flip_prog_list = pickle.load(open('data/df_gpn_flip_prog_list', 'rb'))


## FUNCTIONS
def gen_d_list(tar):
	#
	tars1 = tar_list
	#
	uni0 = id_uni_list[tar_list.tolist().index(tar)] #df_gpn.loc[tar, 'id_unique']
	# prog0 = df_enumP.loc[uni0, 'program']
	prog0 = flip_prog_list[tar_list.tolist().index(tar)]
	d_min_list, part_min_list = [], []
	for tar1 in tars1:
		uni1 = id_uni_list[tar_list.tolist().index(tar1)] #df_gpn.loc[tar1, 'id_unique']
		# prog1 = df_enumP.loc[uni1, 'program']
		prog1 = flip_prog_list[tar_list.tolist().index(tar1)]
		if len(prog1[0])>=len(prog0[0]):
			prog_perm = prog1
			prog_merg = prog0
			uni_merg = uni0
		else:
			prog_perm = prog0
			prog_merg = prog1
			uni_merg = uni1
		part_list = gen_part_list_for_merger_prog(len(prog_merg[0]),len(prog_perm[1]))
		permP_list = gen_perm_prog_array(prog_perm, flipP_included=True) ###
		#
		d_lis, part_lis = [], []
		for part in part_list:
			try: mergP_list = mergP_dict[(uni_merg, part)]
			except KeyError: mergP_list = [prog_merg]
			d_lis += [gen_edit_distance(permP_list, x) for x in mergP_list]
			part_lis += [part for x in mergP_list]
		idx_min = np.argmin(d_lis)
		d_min_list.append(d_lis[idx_min])
		part_min_list.append(part_lis[idx_min])
		# print(tar1, min(d_lis))
	return d_min_list, part_min_list


## MP
if True:
	job_id = int(sys.argv[1]) # 1~45 for 100 tars each
	tars = tar_list[(job_id-1)*100: job_id*100]

	#
	d_lists, part_lists = [], []
	with multiprocessing.Pool(num_cpus) as p:
		jobs = p.map(gen_d_list, tars)
		for job in jobs:
			d_lists.append(job[0])
			part_lists.append(job[1])

	pickle.dump(d_lists, open('data/df_gpn_dprog_arr_%s'%job_id, 'wb'))
	pickle.dump(part_lists, open('data/df_gpn_dprog_part_arr_%s'%job_id, 'wb'))


## assemble jobs (run only after all 45 jobs are done)
if False:
	d_lists, part_lists = [], []
	for job_id in range(1,46):
		ds = pickle.load(open('data/df_gpn_dprog_arr_%s'%job_id,'rb'))
		parts = pickle.load(open('data/df_gpn_dprog_part_arr_%s'%job_id,'rb'))
		d_lists += ds
		part_lists += parts
	d_arr = np.array(d_lists)
	pickle.dump(d_arr, open('data/df_gpn_dprog_arr', 'wb'))
	pickle.dump(part_lists, open('data/df_gpn_dprog_part_arr', 'wb'))


#%% QUICK CHECK
# job_id = 1
# ds = pickle.load(open('data/df_GPN_dprog_arr_%s'%job_id,'rb'))
# parts = pickle.load(open('data/df_GPN_dprog_part_arr_%s'%job_id,'rb'))
# sum([parts[0].count(x) for x in [(1,1),(2,1),(3,1),(2,2),(4,1),(3,2)]])
