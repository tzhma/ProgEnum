'''
note:
1) too many merger groupings to run for prog_size_max>9 (runtime: >10m for 10)
2) if really need to go beyond prog_size>9, consider checking only first few orders of mergers rather than the full list
3) to achieve the above, this line is added: if len(outmap)>8: part_list = [x for x in part_list if len(x) in [len(outmap)-1]]
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
import networkx as nx
import quantecon as qe
from collections import ChainMap
from core_enum import *


## functions: DBs & enumDBs
def gen_fineDBs(dim_u, para):
	'''
	...
	'''
	h,dp,dpm = para
	outmap = (np.arange(dim_u)>=dim_u//2)*2-1 # symmetrical outmap as rewardseek policy

	# knowledge-based discretization with a flexible range for belief value
	inmap_fl_list = []
	inmap_fl_m0_list = []

	alpha = 1
	u_list = np.around(np.linspace((-1+2*h)*alpha, (1-2*h)*alpha, dim_u), 6)

	# inmap_u is a table of transitioned belief values from discretized grid points
	inmap_u = np.zeros([dim_u, 2])
	for i, (u,a) in enumerate(zip(u_list, outmap)):
		for j,o in enumerate([-1,1]):
			u1 = (1-2*h) * (a*o*dp + (1+o*dpm)*u) / (a*o*dp*u + 1+o*dpm + 1e-12)
			inmap_u[i,j] = u1

	# keeping transitioned m0, m1, p0 for later DB enumeration
	inmap_fl_u = inmap_u.flatten() # belief value after transition from grid points
	inmap_fl_m0m1 = [np.argsort(np.abs(x - np.array(u_list)))[:2] for x in inmap_u.flatten()] # two transitioned states closest to the above belief value
	p0_fl = [(u_list[m[1]]-u) / (u_list[m[1]]-u_list[m[0]]) for (m,u) in zip(inmap_fl_m0m1, inmap_fl_u)] # probability of transitioning into closest state
	inmap_fl_m0m1p0 = [(x[0],x[1],y) for x,y in zip(inmap_fl_m0m1, p0_fl)]

	# most likely inmap
	inmap_fl_m0 = tuple([x[y] for x,y in zip(inmap_fl_m0m1, (0,)*(2*dim_u))])
	inmap = np.array(inmap_fl_m0).reshape([len(outmap), 2])
	return outmap, inmap

def gen_DBs_with_discretization_sweep(dim_u, para, m_groups_dict, use_merger_rule):
	'''
	output: valid DBs by discretizating full-B with flexible range of belief values
	note: this is an extension of knowledge-based discretization
	'''
	h,dp,dpm = para
	outmap = (np.arange(dim_u)>=dim_u//2)*2-1 # symmetrical outmap as rewardseek policy

	# belief upper bound
	u_ub = ((1-2*h) + ((1+dpm)/dp*h)**2)**.5 - (1+dpm)/dp*h
	# u_ub = 1-2*h

	# knowledge-based discretization with a flexible range for belief value
	inmap_fl_list = []
	inmap_fl_m0_list = []
	for alpha in np.linspace(.1,1,101):
		u_list = np.around(np.linspace(-u_ub*alpha, u_ub*alpha, dim_u), 6)

		# inmap_u is a table of transitioned belief values from discretized grid points
		inmap_u = np.zeros([dim_u, 2])
		for i, (u,a) in enumerate(zip(u_list, outmap)):
			for j,o in enumerate([-1,1]):
				u1 = (1-2*h) * (a*o*dp + (1+o*dpm)*u) / (a*o*dp*u + 1+o*dpm + 1e-12)
				inmap_u[i,j] = u1

		# keeping transitioned m0, m1, p0 for later DB enumeration
		inmap_fl_u = inmap_u.flatten() # belief value after transition from grid points
		inmap_fl_m0m1 = [np.argsort(np.abs(x - np.array(u_list)))[:2] for x in inmap_u.flatten()] # two transitioned states closest to the above belief value
		p0_fl = [(u_list[m[1]]-u) / (u_list[m[1]]-u_list[m[0]]) for (m,u) in zip(inmap_fl_m0m1, inmap_fl_u)] # probability of transitioning into closest state
		inmap_fl_m0m1p0 = [(x[0],x[1],y) for x,y in zip(inmap_fl_m0m1, p0_fl)]

		# most likely inmap
		inmap_fl_m0 = tuple([x[y] for x,y in zip(inmap_fl_m0m1, (0,)*(2*dim_u))])

		# append
		if inmap_fl_m0 not in inmap_fl_m0_list:
			inmap_fl_list.append(inmap_fl_m0)

	# filter invalid programs (judging only inmap_fl_m0)
	inmap_fl_list = filter_sinks_and_drains(inmap_fl_list, outmap)
	if use_merger_rule: inmap_fl_list = filter_mergers(inmap_fl_list, outmap, m_groups_dict)
	inmap_fl_list = filter_reducible_and_periodic(inmap_fl_list, outmap)
	program_list = gen_program_list_from_inmap_fl_list(inmap_fl_list, outmap)
	return program_list

def gen_DBs_with_discretization_sweep_v2(dim_u, para, m_groups_dict, use_merger_rule):
	'''
	v2: for 3 u_ranges
	'''
	h, dp, dpm = para
	outmap = (np.arange(dim_u)>=dim_u//2)*2-1 # symmetrical outmap as rewardseek policy

	# 5 special belief values
	u_cr = dp/(1-dpm)
	u_ub = ((1-2*h) + ((1+dpm)/dp*h)**2)**.5 - (1+dpm)/dp*h
	u_gl = (1-2*h) * dp/(1-dpm)
	u_gw = (1-2*h) * dp/(1+dpm)
	u_ll = (1-2*h) * (-dp+(1-dpm)*u_ub) / (-dp*u_ub+(1-dpm))

	# knowledge-based discretization with a flexible range for belief value
	inmap_fl_list = []
	inmap_fl_m0_list = []
	u_lists = []

	for alpha in np.linspace(.1,1,101):
		# u ranges
		u_sweep = u_ub * alpha
		u_inner = max(u_gl,u_ll)
		if u_ll<u_gw:
			if u_sweep>u_gw: u_ranges = [(-u_sweep,-u_gw), (-u_inner,u_inner), (u_gw,u_sweep)]
			elif (u_sweep>u_inner) & (u_sweep<=u_gw): u_ranges = [(-u_inner,u_inner)]
			else: u_ranges = [(-u_sweep,u_sweep)]
		else: u_ranges = [(-u_sweep,u_sweep)]

		# u_list
		u_range = sum([x[1]-x[0] for x in u_ranges])
		dim_u_list = [max(2, round(dim_u*(x[1]-x[0])/u_range)) for x in u_ranges]
		if len(dim_u_list)>1: dim_u_list[1] = dim_u - sum([x for i,x in enumerate(dim_u_list) if i!=1])
		if min(dim_u_list)<=0: continue
		u_list = fl([np.linspace(x[0],x[1],y).tolist() for x,y in zip(u_ranges, dim_u_list)])

		# check if u_list is repeated
		if u_list in u_lists: continue
		else: u_lists.append(u_list)

		# inmap_u is a table of transitioned belief values from discretized grid points
		inmap_u = np.zeros([dim_u, 2])
		for i, (u,a) in enumerate(zip(u_list, outmap)):
			for j,o in enumerate([-1,1]):
				u1 = (1-2*h) * (a*o*dp + (1+o*dpm)*u) / (a*o*dp*u + 1+o*dpm + 1e-12)
				inmap_u[i,j] = u1

		# keeping transitioned m0, m1, p0 for later DB enumeration
		inmap_fl_u = inmap_u.flatten() # belief value after transition from grid points
		inmap_fl_m0m1 = [np.argsort(np.abs(x - np.array(u_list)))[:2] for x in inmap_u.flatten()] # two transitioned states closest to the above belief value
		p0_fl = [(u_list[m[1]]-u) / (u_list[m[1]]-u_list[m[0]]) for (m,u) in zip(inmap_fl_m0m1, inmap_fl_u)] # probability of transitioning into closest state
		inmap_fl_m0m1p0 = [(x[0],x[1],y) for x,y in zip(inmap_fl_m0m1, p0_fl)]


		# most likely inmap
		inmap_fl_m0 = tuple([x[y] for x,y in zip(inmap_fl_m0m1, (0,)*(2*dim_u))])

		# append
		if inmap_fl_m0 not in inmap_fl_m0_list:
			inmap_fl_list.append(inmap_fl_m0)

	# filter invalid programs (judging only inmap_fl_m0)
	inmap_fl_list = filter_sinks_and_drains(inmap_fl_list, outmap)
	if use_merger_rule: inmap_fl_list = filter_mergers(inmap_fl_list, outmap, m_groups_dict)
	inmap_fl_list = filter_reducible_and_periodic(inmap_fl_list, outmap)
	program_list = gen_program_list_from_inmap_fl_list(inmap_fl_list, outmap)
	return program_list

def gen_DB_list_with_discretization_sweep(para, m_groups_dict, DB_size_max=20, use_merger_rule=True):
	'''
	this generates DBs with size 2~20
	'''
	program_list = []
	for dim_u in range(2,DB_size_max+1):
		# print('generate DBsize =', dim_u)
		program_list += gen_DBs_with_discretization_sweep(dim_u, para, m_groups_dict, use_merger_rule)
		program_list += gen_DBs_with_discretization_sweep_v2(dim_u, para, m_groups_dict, use_merger_rule)

	## filter out repeated
	program_dict = {(tuple(x[0]), tuple(x[1].flatten())):x for x in program_list}
	program_list_r = list(program_dict.values())
	DBsize_list_r = [len(x[0]) for x in program_list_r]
	idx_sort = np.argsort(DBsize_list_r)
	program_list_r = [program_list_r[x] for x in idx_sort]
	return program_list_r


# ##%% TEST
# m_groups_dict = pickle.load(open('data/m_groups_dict_enumTaskDB','rb'))
# para = .05, .3, -.5
# DB_list = gen_DB_list_with_discretization_sweep(para, m_groups_dict, DB_size_max=5, use_merger_rule=True)
#
# len(DB_list)

#%%
def gen_enumDBs_with_discretization_sweep(dim_u, para, m_groups_dict, p_thres=1):
	h,dp,dpm = para
	outmap = (np.arange(dim_u)>=dim_u//2)*2-1 # symmetrical outmap as rewardseek policy

	# knowledge-based discretization with a flexible range for belief value
	inmap_fl_list = []
	inmap_fl_m0_list = []
	for alpha in np.linspace(.1,1,101):
		u_list = np.around(np.linspace((-1+2*h)*alpha, (1-2*h)*alpha, dim_u), 6)

		# inmap_u is a table of transitioned belief values from discretized grid points
		inmap_u = np.zeros([dim_u, 2])
		for i, (u,a) in enumerate(zip(u_list, outmap)):
			for j,o in enumerate([-1,1]):
				u1 = (1-2*h) * (a*o*dp + (1+o*dpm)*u) / (a*o*dp*u + 1+o*dpm + 1e-12)
				inmap_u[i,j] = u1

		# keeping transitioned m0, m1, p0 for later DB enumeration
		inmap_fl_u = inmap_u.flatten() # belief value after transition from grid points
		inmap_fl_m0m1 = [np.argsort(np.abs(x - np.array(u_list)))[:2] for x in inmap_u.flatten()] # two transitioned states closest to the above belief value
		p0_fl = [(u_list[m[1]]-u) / (u_list[m[1]]-u_list[m[0]]) for (m,u) in zip(inmap_fl_m0m1, inmap_fl_u)] # probability of transitioning into closest state
		p0_fl = np.clip(p0_fl, 0, 1)
		inmap_fl_m0m1p0 = [(x[0],x[1],y) for x,y in zip(inmap_fl_m0m1, p0_fl)]

		# most likely inmap & sum of its probabilities
		inmap_fl_m0 = tuple([x[y] for x,y in zip(inmap_fl_m0m1, (0,)*(2*dim_u))])
		prod_p0 = p0_fl.prod()

		# append
		# if inmap_fl_m0 not in inmap_fl_m0_list:
		inmap_fl_m0_list.append(inmap_fl_m0)
		inmap_fl_list.append((inmap_fl_m0, inmap_fl_m0m1p0, prod_p0))

	# keep the most probable & unrepeated
	id_prod_p0_lists = [[(i,y[-1]) for i,y in enumerate(inmap_fl_list) if y[0]==z] for z in set([x[0] for x in inmap_fl_list])]
	idx_kept_list = []
	for id_prod_p0_list in id_prod_p0_lists:
		idx_top = np.argmax([x[1] for x in id_prod_p0_list])
		idx_kept = id_prod_p0_list[idx_top][0]
		idx_kept_list.append(idx_kept)
	inmap_fl_list = [inmap_fl_list[i] for i in idx_kept_list]

	# enumerate those most probable DBs to cover 50% or 100% likelihood
	inmap_fl_list_enum = []
	for inmap_fl in inmap_fl_list:
		select_list = list(product([0,1], repeat=dim_u*2))
		p_fl_top = [x[-1] for x in inmap_fl[1]]
		p_list = np.array([np.abs(np.prod(np.array(x)-np.array(p_fl_top))) for x in select_list]) # p or 1-p
		p_list = p_list/p_list.sum()
		idx_rnk_list = np.argsort(p_list)[::-1]
		p_cumsum = p_list[idx_rnk_list].cumsum()

		try: idx_thres = (p_cumsum < p_thres).tolist().index(False) + 1 # first index exceeding p_thres; 1e-16 to make it "just below"
		except ValueError: idx_thres = len(p_cumsum)

		select_list_thres = [select_list[i] for i in idx_rnk_list[:idx_thres]]
		if not p_thres: select_list_thres = [select_list[i] for i in idx_rnk_list] # keep all enumDBs
		tmp_list = [tuple([x[y] for x,y in zip(inmap_fl[1], z)]) for z in select_list_thres]
		inmap_fl_list_enum += tmp_list
		print('...',len(tmp_list))

	# filter away invalid DBs
	print('......',len(inmap_fl_list_enum))
	inmap_fl_list_enum = filter_sinks_and_drains(inmap_fl_list_enum, outmap)
	inmap_fl_list_enum = filter_mergers(inmap_fl_list_enum, outmap, m_groups_dict)
	inmap_fl_list_enum = filter_reducible_and_periodic(inmap_fl_list_enum, outmap)
	print('removing repeated DBs with permutations')
	inmap_fl_list_enum = filter_repeats(inmap_fl_list_enum, outmap) # very slow for dim_u>9
	program_list = gen_program_list_from_inmap_fl_list(inmap_fl_list_enum, outmap)
	print(len(program_list))
	return program_list

def gen_enumDB_list_with_discretization_sweep(para, m_groups_dict, DB_size_max=8, p_thres=1):
	'''
	3m to run the case of (DB_size_max=8, p_thres=1-1e-12)
	1h27m to run the case of (DB_size_max=8, p_thres=1-1e-16)
	'''
	program_list = []
	for dim_u in range(2, DB_size_max+1):
		print('dim_u =',dim_u)
		program_list += gen_enumDBs_with_discretization_sweep(dim_u, para, m_groups_dict, p_thres)
	return program_list
