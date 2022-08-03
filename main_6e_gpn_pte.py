'''
...
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
from core_enum import fl
from core_bdp import run_bdp
from core_sdp import run_sdp, gen_pprog_lists, gen_denom_list
from core_pte import gen_perm_prog_array, gen_edit_distance_list, gen_one_merger_prog_list
import multiprocess

num_cpus = int(multiprocess.cpu_count()*1)
print('num_cpus = %s'%num_cpus)


## LOAD
mergP_dict = pickle.load(open('data/mergP_dict','rb'))

df_gpn = pd.read_pickle('data/df_gpn')
tar_list = df_gpn['target'].values
sou_list = df_gpn['source'].values
id_uni_list = df_gpn['id_unique'].values
prog_list = pickle.load(open('data/df_gpn_flip_prog_list', 'rb'))

pprog_arr = pickle.load(open('data/df_gpn_pprog_arr', 'rb'))
dprog_arr = pickle.load(open('data/df_gpn_dprog_arr','rb'))
dprog_part_arr = pickle.load(open('data/df_gpn_dprog_part_arr','rb'))
simprog_arr = pickle.load(open('data/df_gpn_simprog_arr', 'rb'))


## FUNCTIONS: gpn_pte
def gen_df_gpn_1mut():
	'''
	3s
	'''
	query = []
	for i_tar, i_sou in product(range(len(simprog_arr)), repeat=2):
		if i_sou>=i_tar: continue
		if dprog_arr[i_tar,i_sou]!=1: continue
		else:
			tar = tar_list[i_tar]
			sou = tar_list[i_sou]
			part = dprog_part_arr[i_tar][i_sou]
			pp_sim = simprog_arr[i_tar,i_sou]
			eR = df_gpn.loc[tar, 'eR']
			#
			row = tar, sou, eR, part, pp_sim
			query.append(row)

	df_1mut = pd.DataFrame(query, columns=['target', 'source', 'eR', 'part', 'pp_sim'])
	df_1mut = df_1mut.sort_values(['pp_sim', 'eR'], ascending=[False, False])
	return df_1mut

def gen_df_gpn_pte(df_1mut):
	'''
	new pte
	'''
	tar_list_1, query_1 = [], [(1, 0, (1,1), 10)] # add a trivial row for WSLG
	for tar, sou, part, sim in df_1mut[['target', 'source', 'part', 'pp_sim']].values:
		if tar in tar_list_1: continue
		else:
			tar_list_1.append(tar)
			query_1.append((tar, sou, part, sim))

	# df_pte_new
	df_ = pd.DataFrame(query_1, columns=['target', 'source', 'part', 'pp_sim'])
	df_ = df_.sort_values('target', ascending=True)

	# replace source of ogPTE
	df_pte = df_gpn[['id', 'target', 'eR', 'progsize', 'd2enumDB', 'id_unique']]
	df_pte['source'] = df_['source'].values
	df_pte['pp_sim'] = df_['pp_sim'].values
	df_pte['part'] = df_['part'].values

	# other attr
	df_pte['weight'] = [(x!=10)*1+(x==10)*.5 for x in df_['pp_sim'].values]
	z = pprog_arr[range(len(pprog_arr)), range(len(pprog_arr)), -1] * len(pprog_arr)
	z = np.log10(z)
	df_pte['iden'] = z # log inverse ratio
	return df_, df_pte


## RUN: prep
df_1mut = gen_df_gpn_1mut() # 30s
df_, df_pte = gen_df_gpn_pte(df_1mut) # 5s


#%% TEST
df_pte[['target', 'progsize', 'eR', 'source']][:20]
df_[:20]
df_1mut

##%% FUNCTIONS: find branches
def gen_branch_partbranch(tar):
	tar, sou = df_pte[df_pte['target']==tar][['target', 'source']].values[0]

	# append root directly
	roots = [0] # WSLG as the only root
	if tar in roots:
		branch = (tar,)
		partbranch = (tar,)
		return branch, partbranch

	# append tar into branch
	branch = [tar, sou]
	tar_prev = sou
	while tar_prev not in roots:
		tar_curr = branch[-1]
		tar_prev = df_pte[df_pte['target']==tar_curr]['source'].values[0]
		branch.append(tar_prev)
	branch = tuple(branch)

	# convert tar into part
	partbranch = []
	for tar in branch[:-1]:
		part = df_pte[df_pte['target']==tar]['part'].values[0]
		partbranch.append(''.join([str(x) for x in part]))
	partbranch.append(branch[-1])
	partbranch = tuple(partbranch)
	return branch, partbranch

def gen_df_branch_partbranch():
	'''
	multiprocess: reduce tree via grouping nodes according to partitions
	'''
	tar_list = df_pte['target'].values
	branch_list, partbranch_list = [], []
	with multiprocess.Pool(num_cpus) as p:
		jobs = p.map(gen_branch_partbranch, tar_list)
		for prog_id, job in zip(tar_list, jobs):
			branch_list.append(job[0])
			partbranch_list.append(job[1])
			if prog_id % max(tar_list[-1]//100,1)==0: print(prog_id, job)

	# update df_pte
	df_pte['branch'] = [x[::-1] for x in branch_list] # from root to leaf
	df_pte['partbranch'] = [x[::-1] for x in partbranch_list]
	df_pte['d2root'] = [len(x)-1 for x in df_pte['branch']]
	return df_pte

def gen_ecc(prog_id):
	# if df.loc[prog_id, 'if_leaf']==True: ec=0
	if df_pte[df_pte['target']==prog_id]['if_leaf'].values[0]: ec=0
	else:
		outward_branches = []
		for branch in df_pte['branch'].values:
			try: outward_branches.append(branch[branch.index(prog_id)+1:])
			except ValueError: continue
		ec = max([len(x) for x in outward_branches])
	return ec

def gen_df_with_if_leaf():
	'''
	append new column 'if_leaf' in df
	'''
	nonleafnodes = sorted(set(fl([list(x[:-1]) for x in df_pte['branch'].values])))
	df_pte['if_leaf'] = [i not in nonleafnodes for i in df_pte['target'].values]
	return df_pte

def gen_df_ecc():
	'''
	df_pte needs to have 'if_leaf' & 'branch'
	'''
	prog_id_list = df_pte['target'].values
	ecc_list = []
	with multiprocess.Pool(num_cpus) as p:
		jobs = p.map(gen_ecc, prog_id_list)
		for prog_id, job in zip(prog_id_list, jobs):
			ecc_list.append(job)
			if prog_id % max(prog_id_list[-1]//10,1)==0: print(prog_id, job)

	# update df_pte
	df_pte['ecc'] = ecc_list
	return df_pte

def gen_df_tarP_souP():
	'''
	maybe use only in standard embedding rather than the reward-shuffled ones
	'''
	# add first program
	prog0 = prog_list[0] #df_enumP.loc[df_pte.iloc[0]['id_unique'], 'program']
	tar0 = df_pte['target'][0]
	tar0_uni = df_pte[df_pte['target']==tar0]['id_unique'].values[0]
	tarP_dict = {tar0:prog0}
	tarP_list, souP_list = [prog0], [prog0] # first program has no source

	# append lineage
	outmap0, inmap0 = prog0
	prog0_lin = (np.ones_like(outmap0)*tar0_uni, np.ones_like(inmap0)*tar0_uni)
	tarP_lin_dict = {tar0:prog0_lin}
	tarP_lin_list, souP_lin_list = [prog0_lin], [prog0_lin]

	for tar in df_pte['target'].values[1:]:
		if tar % max(len(df_pte)//10,1)==0: print(tar)
		# load
		tar_uni, sou, part = df_pte[df_pte['target']==tar][['id_unique', 'source', 'part']].values[0]
		sou_uni = df_pte[df_pte['target']==sou]['id_unique'].values[0]
		souP = tarP_dict[sou]
		souP_lin = tarP_lin_dict[sou]
		tarP = prog_list[tar_list.tolist().index(tar)] #df_enumP.loc[tar_uni, 'program']

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

	# update df_pte
	df_pte['tarP'] = tarP_list
	df_pte['souP'] = souP_list
	df_pte['tarP_lin'] = tarP_lin_list
	df_pte['souP_lin'] = souP_lin_list

	num_mut_list = []
	for tarP_lin in df_pte['tarP_lin'].values:
		num_mut = (tarP_lin[0]!=4).sum() + (tarP_lin[1]!=4).sum()
		num_mut_list.append(num_mut)
	df_pte['num_mut'] = num_mut_list
	return df_pte


##%% RUN
df_pte = gen_df_branch_partbranch() # 2s
df_pte = gen_df_with_if_leaf()
df_pte = gen_df_ecc()
df_pte = gen_df_tarP_souP() # 20s


##%% FUNCTIONS: iden (top to cover 50% excluding eR<=.25)
df_pte['pp'] = pprog_arr[range(4492),range(4492),-1]
dff = df_pte[df_pte['eR']>.25].sort_values('pp', ascending=False)
num_pp50 = np.where((dff['pp'].cumsum() - dff['pp'].sum()/2)>0)[0][0]
dff = dff.iloc[:num_pp50]
tars_iden = dff['target'].values
tars_all = df_pte[df_pte['eR']>.25].sort_values('pp', ascending=False)['target'].values

# add is_iden
is_iden_list = np.array([(x not in tars_all)*-1 for x in tar_list])
is_iden_list += np.array([(x in tars_iden)*1 for x in tar_list])
df_pte['is_iden'] = is_iden_list

## pickle
col4csv = ['id', 'id_unique', 'target', 'source', 'progsize', 'eR', 'd2enumDB', 'iden', 'd2root', 'num_mut', 'ecc', 'pp_sim', 'weight', 'is_iden']
df_pte[col4csv].to_csv('gephi/df_gpn_pte.csv', index=False)
df_pte.to_pickle('data/df_gpn_pte')
