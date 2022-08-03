'''
run with ox (for higher version of networkx to plot self-loop)
df, df_enumP, mergP_dict has to be globally accessible
mp for 5105: 25s
mp for 268533: 50m
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
from collections import ChainMap
import multiprocessing
from core_enum import fl, reshape_prog, gen_stdz_prog
from core_pte import gen_perm_prog_array, gen_edit_distance_list, gen_one_merger_prog_list

num_cpus = int(multiprocessing.cpu_count()*1)
print('num_cpus = %s'%num_cpus)


## parameters
extension = ('5105', '268533')[1]
progsize_max = (4,5)[1]
filename_load = 'df_pte_' + extension
filename_save = 'df_rpte_' + extension


## load
try: mergP_dict = pickle.load(open('data/mergP_dict','rb'))
except FileNotFoundError: mergP_dict = {}

df_enumP = pd.read_pickle('data/df_enumP_para14')
df = pd.read_pickle('data/%s'%filename_load)
roots = df[df['progsize']==2].index.values


## functions: prep for rPTE with multiprocessing
def gen_branch_partbranch(prog_id):
	prog_id, prog_id_0 = df[df['target']==prog_id][['target', 'source']].values[0]

	# append root directly
	if prog_id in roots:
		branch = (prog_id,)
		partbranch = (prog_id,)
		return branch, partbranch

	# append prog_id into branch
	branch = [prog_id, prog_id_0]
	prog_id_prev = prog_id_0
	while prog_id_prev not in roots:
		prog_id_curr = branch[-1]
		prog_id_prev = df[df['target']==prog_id_curr]['source'].values[0]
		branch.append(prog_id_prev)
	branch = tuple(branch)

	# convert prog_id into part
	partbranch = []
	for prog_id in branch[:-1]:
		part = df[df['target']==prog_id]['part'].values[0]
		partbranch.append(''.join([str(x) for x in part]))
	partbranch.append(branch[-1])
	partbranch = tuple(partbranch)
	return branch, partbranch

def gen_ecc(prog_id):
	# if df.loc[prog_id, 'if_leaf']==True: ec=0
	if df[df['target']==prog_id]['if_leaf'].values[0]: ec=0
	else:
		outward_branches = []
		for branch in df['branch'].values:
			try: outward_branches.append(branch[branch.index(prog_id)+1:])
			except ValueError: continue
		ec = max([len(x) for x in outward_branches])
	return ec

def gen_df_branch_partbranch():
	'''
	multiprocessing: reduce tree via grouping nodes according to partitions
	'''
	prog_id_list = df['target'].values
	branch_list, partbranch_list = [], []
	with multiprocessing.Pool(num_cpus) as p:
		jobs = p.map(gen_branch_partbranch, prog_id_list)
		for prog_id, job in zip(prog_id_list, jobs):
			branch_list.append(job[0])
			partbranch_list.append(job[1])
			if prog_id % max(prog_id_list[-1]//100,1)==0: print(prog_id, job)

	# update df
	df['branch'] = [x[::-1] for x in branch_list] # from root to leaf
	df['partbranch'] = [x[::-1] for x in partbranch_list]
	return

def gen_df_with_if_leaf():
	'''
	append new column 'if_leaf' in df
	'''
	nonleafnodes = sorted(set(fl([list(x[:-1]) for x in df['branch'].values])))
	df['if_leaf'] = [i not in nonleafnodes for i in df['target'].values]
	return

def gen_df_ecc():
	'''
	df needs to have 'if_leaf' & 'branch'
	'''
	prog_id_list = df['target'].values
	ecc_list = []
	with multiprocessing.Pool(num_cpus) as p:
		jobs = p.map(gen_ecc, prog_id_list)
		for prog_id, job in zip(prog_id_list, jobs):
			ecc_list.append(job)
			if prog_id % max(prog_id_list[-1]//10,1)==0: print(prog_id, job)

	# update df
	df['ecc'] = ecc_list
	return

def gen_df_tarP_souP():
	'''
	maybe use only in standard embedding rather than the reward-shuffled ones
	'''
	# add first program
	prog0 = df_enumP.loc[df.iloc[0]['id_unique'], 'program']
	tar0 = df['target'][0]
	tar0_uni = df[df['target']==tar0]['id_unique'].values[0]
	tarP_dict = {tar0:prog0}
	tarP_list, souP_list = [prog0], [prog0] # first program has no source

	# append lineage
	outmap0, inmap0 = prog0
	prog0_lin = (np.ones_like(outmap0)*tar0_uni, np.ones_like(inmap0)*tar0_uni)
	tarP_lin_dict = {tar0:prog0_lin}
	tarP_lin_list, souP_lin_list = [prog0_lin], [prog0_lin]

	for tar in df['target'].values[1:]:
		if tar % max(len(df)//10,1)==0: print(tar)
		# load
		tar_uni, sou, part = df[df['target']==tar][['id_unique', 'source', 'part']].values[0]
		sou_uni = df[df['target']==sou]['id_unique'].values[0]
		souP = tarP_dict[sou]
		souP_lin = tarP_lin_dict[sou]
		tarP = df_enumP.loc[tar_uni, 'program']

		# get minP for both sou and tar
		tar_permP_arr = gen_perm_prog_array(tarP)
		try: sou_mergP_list = mergP_dict[(sou_uni, part)] # sou_mergP has stdz outmap
		except KeyError:
			# sou_mergP_list = [souP + (tuple(range(len(souP[0]))),)]
			sou_mergP_list = gen_one_merger_prog_list(souP, part)
			if sum(part)!=len(souP[0]): mergP_dict[(sou_uni, part)] = sou_mergP_list

		# sou_mergP_list = gen_one_merger_prog_list(souP, part)
		# if sum(part)!=len(souP[0]): mergP_dict[(sou_uni, part)] = sou_mergP_list

		d_list, min_permP_list = gen_edit_distance_list(tar_permP_arr, sou_mergP_list)
		# print(min(d_list))
		idx_min = np.argmin(d_list)
		min_tarP = min_permP_list[idx_min]
		min_souP = sou_mergP_list[idx_min][:2] # 3rd index is linmap
		min_tarP = (np.array(min_tarP[0]), np.reshape(min_tarP[1], [len(min_tarP[0]), 2]))
		min_souP = (np.array(min_souP[0]), np.reshape(min_souP[1], [len(min_souP[0]), 2]))

		# compute lineage
		linmap = sou_mergP_list[idx_min][-1]
		sou_mergP_lin = np.array([souP_lin[0][x] for x in linmap]), np.array([souP_lin[1][x].tolist() for x in linmap]) # expend souP_lin into its mergP_lin
		outmap_lin = (min_tarP[0]==min_souP[0])*sou_mergP_lin[0] + (min_tarP[0]!=min_souP[0])*tar_uni
		inmap_lin = (min_tarP[1]==min_souP[1])*sou_mergP_lin[1] + (min_tarP[1]!=min_souP[1])*tar_uni
		tarP_lin = (outmap_lin, inmap_lin)

		# append
		tarP_dict[tar] = min_tarP
		tarP_lin_dict[tar] = tarP_lin
		tarP_list.append(min_tarP)
		souP_list.append(min_souP)
		tarP_lin_list.append(tarP_lin)
		souP_lin_list.append(sou_mergP_lin)

	# update df
	df['tarP'] = tarP_list
	df['souP'] = souP_list
	df['tarP_lin'] = tarP_lin_list
	df['souP_lin'] = souP_lin_list

	num_mut_list = []
	for tarP_lin in df['tarP_lin'].values:
		num_mut = (tarP_lin[0]!=4).sum() + (tarP_lin[1]!=4).sum()
		num_mut_list.append(num_mut)
	df['num_mut'] = num_mut_list
	return


## functions: rPTE
def gen_part_and_branch_dict():
	'''
	a unique branch is a unique sequence of partition + eccentricity with consistent souP
	'''
	# part id dict is simple
	part_id_dict = {x:i for i,x in enumerate(sorted(set(df['part'].values)))}

	# initialize for branch_dicts
	branch_proglist_dict = {x+(0,):[] for x in sorted(set([tuple(x) for x in df[['partbranch', 'ecc']].values]))}
	branch_source_dict = {x+(0,):() for x in sorted(set([tuple(x) for x in df[['partbranch', 'ecc']].values]))}

	# append first program
	tar, sou, ptb, ecc = df[['target', 'source', 'partbranch', 'ecc']].values[0]
	sou_ptb, sou_ecc = df[df['target']==sou][['partbranch', 'ecc']].values[0]
	branch_proglist_dict[(ptb, ecc, 0)].append(tar)
	branch_source_dict[(ptb, ecc, 0)] = (sou_ptb, sou_ecc, 0)

	# append prog_id
	for tar, sou, ptb, ecc in df[['target', 'source', 'partbranch', 'ecc']].values[1:]:
		# tar, sou, ptb, ecc = df[['target', 'source', 'partbranch', 'ecc']].values[1] #loop[1]
		sou_ptb, sou_ecc = df[df['target']==sou][['partbranch', 'ecc']].values[0]

		# tar, sou, ptb, ecc, sou_ptb, sou_ecc
		# sou in branch_proglist_dict[(sou_ptb, sou_ecc, 0)]

		# find souP's branch
		for i in range(1000000):
			sou_ptb_ext = (sou_ptb, sou_ecc, i)
			if sou in branch_proglist_dict[sou_ptb_ext]: break

		# assign tarP's branch
		for i in range(1000000):
			tar_ptb_ext = (ptb, ecc, i)

			try:
				sou_ptb_ext_0 = branch_source_dict[tar_ptb_ext]
				if len(sou_ptb_ext_0)>0: # having source means tar_ptb_ext is registered
					if sou_ptb_ext_0==sou_ptb_ext: # same source means current tar_ptb_ext is consist
						branch_proglist_dict[tar_ptb_ext].append(tar)
						break
				else: # empty slot to be used
					try:
						branch_proglist_dict[tar_ptb_ext].append(tar)
						branch_source_dict[tar_ptb_ext] = sou_ptb_ext
						break
					except KeyError:
						branch_proglist_dict[tar_ptb_ext] = []
						branch_proglist_dict[tar_ptb_ext].append(tar)
						branch_source_dict[tar_ptb_ext] = sou_ptb_ext
						break

			except KeyError: # tar_ptb_ext not registered yet
				try:
					branch_proglist_dict[tar_ptb_ext].append(tar)
					branch_source_dict[tar_ptb_ext] = sou_ptb_ext
					break
				except KeyError:
					branch_proglist_dict[tar_ptb_ext] = []
					branch_proglist_dict[tar_ptb_ext].append(tar)
					branch_source_dict[tar_ptb_ext] = sou_ptb_ext
					break

	# sort keys & append
	key_sorted = sorted(branch_source_dict)
	branch_source_dict = {x:branch_source_dict[x] for x in key_sorted}
	branch_proglist_dict = {x:branch_proglist_dict[x] for x in key_sorted}
	branch_id_dict = {x:i for i,x in enumerate(key_sorted)}
	branch_tar_sou_dict = {branch_id_dict[x]:branch_id_dict[branch_source_dict[x]] for x in key_sorted}
	branch_tar_proglist_dict = {branch_id_dict[x]:branch_proglist_dict[x] for x in key_sorted}

	# create prog_id_branch_id_dict
	prog_id_branch_id_dict = {}
	for branch_id, proglist in branch_tar_proglist_dict.items():
		for prog_id in proglist:
			prog_id_branch_id_dict[prog_id] = branch_id
	prog_id_branch_id_dict = {x:prog_id_branch_id_dict[x] for x in sorted(prog_id_branch_id_dict)}

	return part_id_dict, branch_id_dict, branch_tar_sou_dict, branch_tar_proglist_dict, prog_id_branch_id_dict

def gen_df_rpte():
	'''
	generate:
	1) df_rpte (index: branch_id)
	2) df_rpte_part_id_dict
	3) df_rpte_branch_id_dict
	'''
	# dict for mapping prog_id to branch_id
	branch_dicts = gen_part_and_branch_dict()
	part_id_dict, branch_id_dict, branch_tar_sou_dict, branch_tar_proglist_dict, prog_id_branch_id_dict = branch_dicts

	# collect branches to df
	query = []
	print(len(branch_id_dict), 'partbranches to be process...')
	for (partbranch, ecc, ext), tar in branch_id_dict.items():
		print(partbranch)

		if len(partbranch)==1: part_id = 0
		else: part_id = part_id_dict[tuple([int(x) for x in partbranch[-1]])]

		sou = branch_tar_sou_dict[tar]
		prog_ids = branch_tar_proglist_dict[tar]

		# read other prog attr
		unique_ids, progsizes, eRs = np.array([df[df['target']==x][['id_unique','progsize','eR']].values[0] for x in prog_ids]).T
		unique_ids = unique_ids.astype('int')
		progsize = progsizes.mean().astype('int')
		eR_mean, eR_std = eRs.mean(), eRs.std()
		num_prog = len(prog_ids)
		deg = len(partbranch)-1

		# append
		query.append([tar, sou, tar, part_id, unique_ids, prog_ids, eRs, progsize, eR_mean, eR_std, num_prog, deg, ecc])

	# df
	df_rpte = pd.DataFrame(query, columns=['id', 'source', 'target', 'part_id', 'unique_ids', 'prog_ids', 'eRs', 'progsize', 'eR_mean', 'eR_std', 'num_prog', 'deg', 'ecc'])

	# dict_r = {v:''.join([str(x) for x in k]) for (k,v) in part_id_dict.items()}
	# pd.DataFrame.from_dict(dict_r, orient='index', columns=['part']).to_csv('gephi/%s_part_dict.csv'%filename_save, index=False)

	# df_rpte.to_csv('gephi/%s.csv'%filename_save, index=False)
	# dict_r = {v:k for (k,v) in branch_id_dict.items()}
	# pd.DataFrame.from_dict(dict_r, orient='index').to_csv('gephi/%s_branch_dict.csv'%filename_save, index=False)

	return df_rpte, part_id_dict, branch_id_dict, branch_tar_sou_dict, branch_tar_proglist_dict, prog_id_branch_id_dict

def gen_rpte_branch(tar, roots=[0]):
	tar, sou = df_rpte[df_rpte['target']==tar][['target', 'source']].values[0]
	if tar in roots:
		branch = (tar,)
		return branch

	# append prog_id into branch
	branch = [tar, sou]
	tar_prev = sou
	while tar_prev not in roots:
		tar_curr = branch[-1]
		tar_prev = df_rpte[df_rpte['target']==tar_curr]['source'].values[0]
		branch.append(tar_prev)
	branch = tuple(branch)
	return branch

def gen_df_rpte_branch():
	'''
	eff_rpte is a reward filtered subtree
	'''
	tar_list = df_rpte['target'].values
	branch_list = []
	for tar in tar_list:
		branch = gen_rpte_branch(tar)
		branch_list.append(branch)

	# update df & save
	df_rpte['branch'] = [x[::-1] for x in branch_list] # from root to leaf
	return


## functions: good program network (GPN)
def gen_df_rpte_good_prog_net(eR_thres=0.277911, progsize_max=5):
	'''
	note: run this after loading:
	part_id_dict, branch_id_dict, branch_tar_sou_dict, branch_tar_proglist_dict, prog_id_branch_id_dict = pickle.load(open('data/df_rpte_%s_branch_dicts'%extension,'rb'))

	a single-component good program network extracted from df_rpte
	df: e.g., df_rpte_5015_branch
	df_rpte: e.g., df_rpte_5015
	'''
	# set df progsize upperbound
	dff = df[df['progsize']<=progsize_max]

	# find GPN
	tar_list = dff[dff['eR']>=eR_thres]['target'].values
	idx_list = dff[dff['eR']>=eR_thres].index
	tar_all_list = sorted(set(fl(dff.loc[idx_list, 'branch'])))
	tar_conn_list = [x for x in tar_all_list if x not in tar_list]
	eR_all_list = [dff[dff['target']==x]['eR'].values[0] for x in tar_all_list]

	# plot: eR_hist
	plt.figure(figsize=(10,4), dpi=300)
	plt.title(np.sort(eR_all_list)[:10])
	plt.hist(eR_all_list, bins=100)
	if not eR_thres==0.277911: plt.axvline(x=eR_thres, color='r', linestyle='dashed')
	plt.axvline(x=0.277911, color='r', linestyle='dashed')
	plt.axvline(x=0.263867, color='r', linestyle='dashed')
	plt.axvline(x=0.25, color='r', linestyle='dashed')
	plt.xlabel('eR')
	plt.show()

	# next: which compound nodes do they belong?
	compound_list = [prog_id_branch_id_dict[x] for x in tar_list]
	compound_all_list = [prog_id_branch_id_dict[x] for x in tar_all_list]
	compound_conn_list = [prog_id_branch_id_dict[x] for x in tar_conn_list]
	compound_set = sorted(set(compound_list))
	compound_all_set = sorted(set(compound_all_list))
	compound_conn_set = sorted(set(compound_conn_list))

	count_dict = {x:compound_list.count(x) for x in compound_set}
	count_all_dict = {x:compound_all_list.count(x) for x in compound_all_set}
	count_conn_dict = {x:compound_conn_list.count(x) for x in compound_conn_set}

	# add other GPN attr in df_rpte
	GPN_prog_ids_list, GPN_eRs_list = [], []
	GPN_eR_mean_list,   GPN_eR_std_list = [], []
	GPN_num_prog_list = []
	GPN_conn_prog_ids_list = []
	for compound_tar in df_rpte['target'].values:
		GPN_prog_ids = [x for x in df_rpte[df_rpte['target']==compound_tar]['prog_ids'].values[0] if x in tar_all_list]
		GPN_conn_prog_ids = [x for x in df_rpte[df_rpte['target']==compound_tar]['prog_ids'].values[0] if x in tar_conn_list]
		GPN_eRs = [dff[dff['target']==x]['eR'].values[0] for x in GPN_prog_ids]
		if len(GPN_eRs)==0:
			GPN_eR_mean = 0
			GPN_eR_std = 0
			GPN_num_prog = 0
		else:
			GPN_eR_mean, GPN_eR_std = np.mean(GPN_eRs), np.std(GPN_eRs)
			GPN_num_prog = len(GPN_prog_ids)

		# append
		GPN_prog_ids_list.append(GPN_prog_ids)
		GPN_conn_prog_ids_list.append(GPN_conn_prog_ids)
		GPN_eRs_list.append(GPN_eRs)
		GPN_eR_mean_list.append(GPN_eR_mean)
		GPN_eR_std_list.append(GPN_eR_std)
		GPN_num_prog_list.append(GPN_num_prog)
	#
	df_rpte['if_GPN'] = [(x in compound_all_set)*1 for x in df_rpte['target'].values]
	df_rpte['GPN_prog_ids'] = GPN_prog_ids_list
	df_rpte['GPN_eRs'] = GPN_eRs_list
	df_rpte['GPN_eR_mean'] = GPN_eR_mean_list
	df_rpte['GPN_eR_std'] = GPN_eR_std_list
	df_rpte['GPN_num_prog'] = GPN_num_prog_list
	df_rpte['if_GPN_conn'] = [(x in compound_conn_set)*1 for x in df_rpte['target'].values]
	df_rpte['GPN_conn_prog_ids'] = GPN_conn_prog_ids_list

	# extract GPN data
	dff_rpte = df_rpte[[x in compound_all_set for x in df_rpte['target'].values]]
	branch_all_list = sorted(dff_rpte['branch'].values)
	lineage_list = []
	for branch_0, branch in zip(branch_all_list[:-1], branch_all_list[1:]):
		if branch[:-1]!=branch_0: lineage_list.append(branch_0)

	# print
	print('%s programs are needed for covering %s programs with eR>=eR_WSLG'%(len(tar_all_list), len(tar_list)))
	print('%s compound nodes are needed for covering %s compound nodes with eR>=eR_WSLG'%(len(compound_all_set), len(compound_set)))
	print('lineage:')
	print(lineage_list)

	return lineage_list, branch_all_list

def gen_df_rpte_d2enumDB():
	dff = df.copy()
	dff.index = dff['target'].values

	# df d2enumDB
	d2enumDB_list = []
	GPN_d2enumDB_list = []
	for progids, GPN_progids in df_rpte[['prog_ids','GPN_prog_ids']].values:
		#
		d2enumDBs = dff.loc[progids, 'd2enumDB'].values
		if d2enumDBs.size==0: d2enumDB = 0
		else: d2enumDB = np.around(d2enumDBs.mean(), 6)
		d2enumDB_list.append(d2enumDB)
		#
		GPN_d2enumDBs = dff.loc[GPN_progids, 'd2enumDB'].values
		if GPN_d2enumDBs.size==0: GPN_d2enumDB = 0
		else: GPN_d2enumDB = np.around(GPN_d2enumDBs.mean(), 6)
		GPN_d2enumDB_list.append(GPN_d2enumDB)
	#
	df_rpte['d2enumDB'] = d2enumDB_list
	df_rpte['GPN_d2enumDB'] = GPN_d2enumDB_list

	# df d2_enumDB_branches
	d2enumDB_branch_list = []
	GPN_d2enumDB_branch_list = []
	GPN_d2enumDB_branch_list_4 = []
	for compound_id in df_rpte.index:
		d2enumDB_branch = df_rpte.loc[list(df_rpte.loc[compound_id, 'branch']), 'd2enumDB'].values.tolist()
		d2enumDB_branch_list.append(d2enumDB_branch)
		GPN_d2enumDB_branch = df_rpte.loc[list(df_rpte.loc[compound_id, 'branch']), 'GPN_d2enumDB'].values.tolist()
		GPN_d2enumDB_branch_list.append(GPN_d2enumDB_branch)
	#
	df_rpte['d2enumDB_branch'] = d2enumDB_branch_list
	df_rpte['GPN_d2enumDB_branch'] = GPN_d2enumDB_branch_list
	return


##%% run: rPTE
if True:
	# append new column to df
	gen_df_branch_partbranch()
	gen_df_with_if_leaf()
	gen_df_ecc()
	gen_df_tarP_souP()

	# collecting branches and reduce tree
	df_rpte, part_id_dict, branch_id_dict, branch_tar_sou_dict, branch_tar_proglist_dict, prog_id_branch_id_dict = gen_df_rpte()
	gen_df_rpte_branch() # add branch
	lineage_list, branch_all_list = gen_df_rpte_good_prog_net(progsize_max=progsize_max)
	gen_df_rpte_d2enumDB() # add d2enumDB for compound nodes

	all_dicts = 'part_id_dict, branch_id_dict, branch_tar_sou_dict, branch_tar_proglist_dict, prog_id_branch_id_dict, lineage_list, branch_all_list', part_id_dict, branch_id_dict, branch_tar_sou_dict, branch_tar_proglist_dict, prog_id_branch_id_dict, lineage_list, branch_all_list

	# pickle
	if not os.path.exists('data'): os.makedirs('data')
	if not os.path.exists('gephi'): os.makedirs('gephi')

	pd.to_pickle(df_rpte, 'data/%s'%filename_save)
	df_rpte.to_csv('gephi/%s.csv'%filename_save, index=False)

	pickle.dump(mergP_dict, open('data/mergP_dict_%s'%extension, 'wb'))
	pickle.dump(all_dicts, open('data/df_rpte_%s_alldicts'%extension, 'wb'))
