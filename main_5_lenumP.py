'''
global variables:
1) df: a master program list during morphing
2) df_enumP
3) mergP_dict

note: never use edit_distance if only to check if identical... too slow!
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
from collections import ChainMap
from core_enum import fl, gen_inv_perm_list, gen_prog_fl_list, filter_mergers, filter_sinks_and_drains, filter_reducible_and_periodic, filter_repeats, perm_inmap_fl, gen_perm_standard, group_prog_by_outmap


if not os.path.exists('plot'): os.makedirs('plot')


## PARAMETERS
progsize_max, fac_log2, num_gen = 4, .5, 12
# progsize_max, fac_log2, num_gen = 5, 10, 16


## LOAD
try:
	m_groups_dict = pickle.load(open('data/m_groups_dict_enumP','rb'))
except FileNotFoundError:
	m_groups_dict = gen_m_groups_dict(prog_size_max=8)
	pickle.dump(m_groups_dict, open('data/m_groups_dict_enumP', 'wb'))

df_gpn = pd.read_pickle('data/df_gpn')
df_enumP = pd.read_pickle('data/df_enumP_para14')
df_enumP_sort = pd.read_pickle('data/df_enumP_para14_sorted')
df_enumDB = pd.read_pickle('data/df_enumDB_para14')
mergP_dict = pickle.load(open('data/mergP_dict', 'rb'))


## FUNCTIONS
def gen_df_init():
	id_unique_select = [3, 4] # P2s with eR>eR_rand
	df = df_enumP.loc[id_unique_select,['program', 'progsize', 'eR', 'd2enumDB']]
	df[['id_unique', 'target']] = np.array([df.index, df.index]).T

	# add columns
	df[['source', 'part_born']] = None
	df['t_born'] = 0
	df['t_morph'] = [[]]*2
	df['part_morph'] = [[]]*2
	df['status'] = 'morph' # 'idle', 'morph', 'done'

	# reindex
	df.index = range(len(id_unique_select))
	return df

def gen_mutants(program):
	'''
	note: also works for prog_fl
	1-mutation:
	1) choose one entry out of 3*progsize
	2) if the chosen is outmap entry: flip the sign
	3) if the chosen is inmap entry: choose m1 out of range(progsize) excl m
	4) standardize mutant outmaps by flipping sign + permuting nodes
	5*) apply rule-out rules to mutants except permutation-rule (since later will check again in append_op)
	'''
	# load
	prog_fl = gen_prog_fl_list([program])[0]
	progsize = len(program[0])

	mutant_fl_list = []
	for m in range(len(prog_fl)):
		if m<progsize: # change outmap
			mutant_fl = list(prog_fl)
			mutant_fl[m] = -mutant_fl[m]
			if not len(set(mutant_fl[:progsize]))==1:
				mutant_fl_list.append(mutant_fl) # only append when both actions are in outmap
		else: # change inmap
			for m1 in [x for x in range(progsize) if x!=mutant_fl[m]]:
				mutant_fl = list(prog_fl)
				mutant_fl[m] = m1
				mutant_fl_list.append(mutant_fl)

	# standardize outmaps
	mutant_fl_list_1 = []
	for mutant_fl in mutant_fl_list:
		outmap, inmap_fl = tuple(mutant_fl[:progsize]), tuple(mutant_fl[progsize:])
		perm, outmap1 = gen_perm_standard(outmap)
		inmap1_fl = perm_inmap_fl(inmap_fl, perm)
		mutant_fl_list_1.append((outmap1, inmap1_fl))

	# group mutants by outmap
	mutant_fl_dict = group_prog_by_outmap(mutant_fl_list_1)

	# filter
	filtered_list = []
	for outmap, inmap_fl_list in mutant_fl_dict.items():
		inmap_fl_list = filter_sinks_and_drains(inmap_fl_list, outmap)
		inmap_fl_list = filter_mergers(inmap_fl_list, outmap, m_groups_dict)
		inmap_fl_list = filter_reducible_and_periodic(inmap_fl_list, outmap)
		inmap_fl_list = filter_repeats(inmap_fl_list, outmap)
		filtered_list += [(outmap, x) for x in inmap_fl_list]
	return filtered_list

def morph_op(progsize_max=5):
	'''
	1) grow at most one state
	2) 1-mutation
	input: load programs with status='morph'
	output: mutant_dict (target; key:(source, part_born), value:prog_fl_list)
	'''
	# df_1 for avoiding UnboundLocalError
	df_1 = df.copy()

	## assign 't_morph', 'part_morph'
	idx_morph = df_1[(df_1['status']=='morph')].index.values

	t_morph = df['t_born'].max() # read this gloabally
	t_morph_list = [x+[t_morph] for x in df_1.loc[idx_morph, 't_morph']]
	part_morphs_list = []
	for progsize, part_morphs in df.loc[idx_morph, ['progsize', 'part_morph']].values:
		if progsize<progsize_max:
			part_morph = ((1,)*progsize, (2,)+(1,)*(progsize-1)) # two most simple morphs
		else:
			part_morph = ((1,)*progsize, ) # only simple morphs if progsize_max
		part_morphs_list.append(part_morphs + [part_morph])

	# update df
	df_1.loc[idx_morph, 't_morph'] = t_morph_list
	df_1.loc[idx_morph, 'part_morph'] = part_morphs_list
	df_1.index = range(len(df_1))

	## create mutants
	mutant_dict = {} # key:(source, part_born), value:prog_fl_list
	for prog_id in idx_morph:
		# load
		id_unique, progsize, program = df_1.loc[prog_id, ['id_unique', 'progsize', 'program']]
		part_morphs = df_1.loc[prog_id, 'part_morph'][-1] # last morph in history is to be executed

		# find mutants
		for part in part_morphs:
			# load
			if sum(part)>progsize: prog_fl_list = mergP_dict[(id_unique, part)] # load mergPs
			else: prog_fl_list = [program] # program list also compatible

			# generate mutants
			mutant_fl_list = []
			for prog_fl in prog_fl_list: mutant_fl_list += gen_mutants(prog_fl)

			# append
			mutant_dict[(id_unique, part)] = mutant_fl_list

	# filter repeats
	mutant_group_dict = group_prog_by_outmap(fl(mutant_dict.values()))
	filtered_list = []
	for outmap, inmap_fl_list in mutant_group_dict.items():
		inmap_fl_list = filter_repeats(inmap_fl_list, outmap)
		filtered_list += [(outmap, x) for x in inmap_fl_list]

	# update mutant_dict
	mutant_dict_1 = {}
	for k,v in mutant_dict.items():
		mutant_dict_1[k] = [x for x in v if x in filtered_list]
	return df_1, mutant_dict_1

def append_and_eval_op(mutant_dict):
	'''
	1) input: mutant_dict (target; key:(source, part_born), value:prog_fl_list)
	2) identify id_unique for each mutant
	3) append only non-existent id_unique to df
	4) change status: 'morph'->'done'
	'''
	# df_1 for avoiding UnboundLocalError
	df_1 = df.copy()

	# hash all relevant programs (for finding id_unique)
	idx_dict = {}
	inmap_fl_list = [tuple(x[1].flatten()) for x in df_enumP['program'].values]
	idx_list = df_enumP.index.values
	for idx, inmap_fl_0 in zip(idx_list, inmap_fl_list): idx_dict[inmap_fl_0] = idx

	# find unique newborns
	for (source, part), prog_fl_list in mutant_dict.items():
		# print('appending source, part = %s, %s'%(source, part))

		id_unique_newborns = []
		for prog_fl in prog_fl_list:
			outmap, inmap_fl = prog_fl
			progsize = len(outmap)

			# permute inmaps
			inv_perm_list = gen_inv_perm_list(outmap, keep_unpermuted=True)
			inmap_fl_perm_list = [perm_inmap_fl(inmap_fl, x) for x in inv_perm_list]

			# find id_unique from dict
			for inmap_fl_perm in inmap_fl_perm_list:
				try:
					id_unique_0 = idx_dict[inmap_fl_perm]
					if id_unique_0 not in df_1['id_unique'].values:
						id_unique_newborns.append(id_unique_0)
					break
				except KeyError: continue

		# append newborns to df
		id_unique_newborns = sorted(set(id_unique_newborns))
		num_newborns = len(id_unique_newborns)
		t_newborns = df['t_born'].max() + 1  # read this gloabally
		program_newborns = df_enumP.loc[id_unique_newborns, 'program'].values
		progsize_newborns = df_enumP.loc[id_unique_newborns, 'progsize'].values
		eR_newborns = df_enumP.loc[id_unique_newborns, 'eR'].values
		d2enumDB_newborns = df_enumP.loc[id_unique_newborns, 'd2enumDB'].values
		#
		df_newborns = pd.DataFrame([program_newborns, progsize_newborns, eR_newborns, d2enumDB_newborns, id_unique_newborns, id_unique_newborns, [source]*num_newborns, [part]*num_newborns, [t_newborns]*num_newborns, [[]]*num_newborns, [[]]*num_newborns, ['idle']*num_newborns]).T
		df_newborns.columns = df_1.columns.values
		#
		df_1 = pd.concat([df_1, df_newborns])

	# change status
	df_1.index = range(len(df_1))
	idx_morph = df_1[df_1['status']=='morph'].index.values
	df_1.loc[idx_morph, 'status'] = 'done'
	return df_1

def select_op(eR_thres=.263867, type_select='top_log2', fac_log2=10):
	'''
	choose to be morphed based on 'idle' &
	1) eR > eR_thres
	2) top log2(#idlers)
	update df: 't_morph', 'part_morph'
	change status: 'idle'->'morph'
	'''
	# df_1 for avoiding UnboundLocalError
	df_1 = df.copy()

	# change status
	if type_select=='eR_thres':
		idx_morph = df_1[(df_1['status']=='idle') & (df_1['eR']>=eR_thres)].index.values

	if type_select=='top_log2':
		dff = df_1[(df_1['status']=='idle')].sort_values('eR', ascending=False)
		num_top = np.ceil(np.log2(len(dff)) * fac_log2).astype('int')
		idx_morph = dff[:num_top].index

	df_1.loc[idx_morph, 'status'] = 'morph'
	# print(len(idx_morph), 'selected to be morphed')
	return df_1


## RUN
if False:
	df = gen_df_init()

	num_prog_list = [len(df)]
	for t in range(num_gen):
		df, mutant_dict = morph_op(progsize_max=progsize_max)
		df = append_and_eval_op(mutant_dict)
		df = select_op(fac_log2=fac_log2)
		num_prog_list.append(len(df))
		print(num_prog_list[-5:])

		# plot
		num_enumP = len(df_enumP[df_enumP['progsize']<=progsize_max])
		plt.figure(figsize=(10,4), dpi=300)
		plt.title('t = %s, #prog_lenum = %s, #prog_enum = %s'%(t, num_prog_list[-5:], num_enumP))
		plt.hist(df_enumP[df_enumP['progsize']<=progsize_max]['eR'], bins=100, label='full enumP')
		plt.hist(df[df['progsize']<=progsize_max]['eR'], bins=100, label='local enumP')
		plt.hist(df_enumDB[df_enumDB['progsize']<=progsize_max]['eR'], bins=100, label='enumDB')
		plt.xlabel('eR')
		plt.legend()
		plt.savefig('plot/morph_%s'%t)

		# print
		enumPs = df_enumP[df_enumP['progsize']<=progsize_max].index.values
		dff = df_gpn[(df_gpn['progsize']<=progsize_max) & (df_gpn['eR']>.277911)]
		gpnPs = dff['id_unique'].values
		lenumPs = df['id_unique'].values
		frac_all = sum([x in enumPs for x in lenumPs])/(len(enumPs)-3)
		frac_gpn = sum([x in gpnPs for x in lenumPs])/len(dff)
		print('iter:%s'%t, progsize_max, frac_all, frac_gpn)

	# pickle
	df.to_pickle('data/df_lenumP%s'%progsize_max)
	df.to_csv('gephi/df_lenumP%s.csv'%progsize_max, index=False)
