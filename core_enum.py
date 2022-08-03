'''
enumCore provides functions commonly used in enumP, enumDB, and lenum`P
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


## functions: global
def fl(l):
	return [x for y in l for x in y]

def reshape_prog(program, output_type='prog_mat'):
	'''
	output_type:
	1) program: (outmap, inmap)
	2) prog_fl: (outmap, inmap_fl)
	3) prog_fl_: outmap + inmap_fl
	4) prog_mat: (outmap, inmap_mat)
	'''
	# load
	if type(program[1])==np.ndarray:
		if len(program[1].shape)==2: outmap, inmap = program
		else: outmap, inmap_mat = program
	elif type(program[1])==tuple:
		outmap, inmap_fl = program
	else:
		progsize = len(program)//3
		outmap, inmap_fl = program[:progsize], program[progsize:]

	# gen all type of inmap
	try:
		# inmap_mat
		idx = np.where(inmap_mat==1)
		inmap = np.zeros([inmap_mat.shape[-1], 2]).astype('int')
		inmap[idx[1],idx[0]] = idx[2]
		inmap_fl = tuple(inmap.flatten())
	except UnboundLocalError:
		try:
			# inmap
			inmap_mat = np.zeros([2, len(inmap), len(inmap)])
			inmap_mat[0, np.arange(len(inmap)), inmap[:,0]] = 1
			inmap_mat[1, np.arange(len(inmap)), inmap[:,1]] = 1
			inmap_fl = tuple(inmap.flatten())
		except UnboundLocalError:
			inmap = np.reshape(inmap_fl, [len(outmap), 2])
			inmap_mat = np.zeros([2, len(inmap), len(inmap)])
			inmap_mat[0, np.arange(len(inmap)), inmap[:,0]] = 1
			inmap_mat[1, np.arange(len(inmap)), inmap[:,1]] = 1

	# output converted
	outmap = tuple(outmap)
	if output_type=='program': return (outmap, inmap)
	if output_type=='prog_fl': return (outmap, inmap_fl)
	if output_type=='prog_fl_': return outmap + inmap_fl
	if output_type=='prog_mat': return (outmap, inmap_mat)
	print('%s is not supported'%output_type)
	return

def gen_prog_fl_list(program_list):
	'''
	also works for prog_fl_list
	'''
	try: prog_fl_list = [tuple(x[0]) + tuple(x[1].flatten()) for x in program_list]
	except AttributeError: prog_fl_list = [x[0] + x[1] for x in program_list]
	return prog_fl_list

def gen_perm_standard(outmap):
	'''
	this generate a permutation that gives standardized outmap
	e.g., (1,-1,1) -> (-1,1,1); perm = (1,0,2)
	e.g., (-1,-1,1) -> (1,1,-1); perm = (2,0,1)
	'''
	if sum(outmap)<0: outmap = -np.array(outmap)
	perm = tuple(np.argsort(outmap))
	outmap1 = tuple(np.sort(outmap))
	return perm, outmap1

def group_prog_by_outmap(program_list):
	'''
	output program_dict with
	key: outmap
	value: inmap_list
	'''
	if len(program_list[0])%3==0:
		progsize = len(program_list[0])//3
		program_list = [(tuple(x[:progsize]), tuple(x[progsize:])) for x in program_list]
	outmap_set = set([x[0] for x in program_list])
	program_dict = {y:[x[1] for x in program_list if x[0]==y] for y in outmap_set}
	return program_dict

def partition(n):
	z = set()
	z.add((n, ))
	for x in range(1, n):
		for y in partition(n - x):
			z.add(tuple(sorted((x, ) + y, reverse=True)))
	return sorted(z, key=len, reverse=True)

def gen_inv_perm_list(outmap, keep_unpermuted=False, keep_flipped=True):
	'''
	generate permutations that
	1) doesn't change outmap
	2) change outmap to -outmap (task is symmetric under policy-inversion)
	'''
	outmap = list(outmap)
	y0 = []
	for perm in list(permutations(range(len(outmap)), len(outmap))):
		y1 = np.array(outmap)[list(perm)].tolist()
		if y1==outmap: y0.append(list(perm))
		elif (len(outmap)%2==0) & (y1==[-x for x in outmap]):
			if keep_flipped: y0.append(list(perm))
		else: pass
	if keep_unpermuted: return y0
	return y0[1:]

def gen_one_outmap_list(prog_size):
	y0 = sorted([sorted(x) for x in partition(prog_size) if len(x)==2])
	y1 = []
	for x in y0:
		y2 = []
		for i, a in enumerate([-1,1]): y2 += [a]*x[i]
		y1.append(tuple(y2))
	return y1

def gen_outmap_list(prog_size_max, for_enumDB=False):
	'''
	for enumP:
	1) each outmap has a unique numbers of a- and a+
	2) an outmap cannot be all a-
	3) num_a- < num_a+
	note: such set of outmaps for enumeration only make sense if world-state transition and the roles of a-, a+ are symmetrical.

	for enumDB:
	generate symmetric outmaps
	'''
	if for_enumDB:
		y = []
		for prog_size in range(2, prog_size_max+1):
			outmap = (np.arange(prog_size)>=prog_size//2)*2-1
			y.append(tuple(outmap))

	else:
		y = []
		for prog_size in range(2, prog_size_max+1):
			y += gen_one_outmap_list(prog_size)
	return y


## functions: merger rule
def gen_one_m_groups_list(part, outmap):
	'''
	1) given a partition; e.g., part=(2,2,1)
	2) generate groups for each partition; e.g., [(0,1), (2,3), (4,)], [(0,2), (1,3), (4,)], ...
	3) remove repeats; e.g., [(0,1), (2,3), (4,)] & [(2,3), (0,1), (4,)] are the same
	'''
	m_groups_list = [[x] for x in combinations(range(len(outmap)), part[0])]
	for i in range(1,len(part)):
		if part[i]==1: break
		m_groups_list_1 = [x for x in m_groups_list if len(x)==i]
		for m_groups in m_groups_list_1:
			par = part[len(m_groups)]
			m_remain = [x for x in range(len(outmap)) if x not in fl(m_groups)]
			cand_list = list(combinations(m_remain, par))
			m_groups_list_2 = []
			for cand in cand_list:
				m_groups_1 = sorted(m_groups + [cand])
				if m_groups_1 not in m_groups_list: m_groups_list_2.append(m_groups_1)
			m_groups_list += m_groups_list_2

	# fill in remaining individual states
	m_groups_list_3 = []
	for m_groups in m_groups_list:
		m_remain = [(x,) for x in range(len(outmap)) if x not in fl(m_groups)]
		m_groups_1 = sorted(m_groups + m_remain)
		m_groups_list_3.append(m_groups_1)

	# remove incomplete groups
	y = [sorted(x, key=len, reverse=True) for x in m_groups_list_3 if len(x)==len(part)]
	return y

def gen_m_groups_list(outmap, m_groups_dict=False):
	'''
	note: if already run gen_m_groups_dict, readout m_groups_list(outmap) from file to save time
	1) generate all partitions of M; e.g. M=5 -> (2,2,1)
	2) generate unique groups for each partition;
			e.g., [(0,1), (2,3), (4,)], [(0,2), (1,3), (4,)], ...
	3) remove partitions that have groups containing different actions
	'''
	if not m_groups_dict:
		part_list = partition(len(outmap))[1:-1]
		if len(outmap)>8: part_list = [x for x in part_list if len(x) in [len(outmap)-1]]
		m_groups_list = []
		for part in part_list:
			m_groups_list += gen_one_m_groups_list(part, outmap)

		# remove groups that contain two actions
		y = []
		for m_groups in m_groups_list:
			a_groups = [[np.array(outmap)[x] for x in y] for y in m_groups]
			if np.prod([len(set(x)) for x in a_groups])==1: y.append(m_groups)
	else: y = m_groups_dict[tuple(outmap)]
	return y

def gen_m_groups_dict(prog_size_max=8, for_enumDB=False):
	'''
	1) this generates all possible grouping for program states m
	2) each group contains only one action
	'''
	outmap_list = gen_outmap_list(prog_size_max, for_enumDB=for_enumDB)
	y = {}
	for x in outmap_list:
		m_groups_list = gen_m_groups_list(x)
		y[x] = m_groups_list
	return y

def if_mergers(inmap_fl, outmap, m_groups_dict):
	'''
	note: this new generalized merger rule checks works for all possible partitions of program states, e.g., M=5 -> [(2,3,4), (0,), (1,)]; while the old merger rule only removes first-order mergers, e.g., M=5 -> [(2,3), (0,), (1,), (4,)]
	1) for each grouping, map old states (before merging) to new states (after merging);
		e.g, m_groups=[(2,3,4), (0,), (1,)] -> map={2:A, 3:A, 4:A, 0:B, 1:C}
	2) relabel states in inmap;
		e.g., inmap=[[1,0],[0,2],[1,3],[1,4],[1,2]] -> inmap1=[[C,B],[B,A],[C,A],[C,A],[C,A]]
	3) check if each group in inmap has consistent winning & losing transitions;
		for group (2,3,4): inmap1[(2,3,4),:] = [[C,A],[C,A],[C,A]] has consistent losing transition: C & consistent winning transition: A
	'''
	m_groups_list = gen_m_groups_list(outmap, m_groups_dict)
	# go thru m_groups, and stop once mergers are found
	for m_groups in m_groups_list:
		# relabel states based on grouping
		m_map = dict(ChainMap(*[{x:i for x in y} for i,y in enumerate(m_groups)]))
		inmap1 = np.reshape([m_map[x] for x in inmap_fl], [len(outmap),2])

		# check if winning & losing columns of a group contains only one mergered state
		m_groups_after_map = fl([(inmap1[x,:][:,0], inmap1[x,:][:,1]) for x in m_groups if len(x)>1])
		if_mergers = np.prod([len(set(x)) for x in m_groups_after_map])==1
		if if_mergers: return if_mergers
	return False

def filter_mergers(inmap_fl_list, outmap, m_groups_dict):
	filtered_list = []
	for i, inmap_fl in enumerate(inmap_fl_list):
		if if_mergers(inmap_fl, outmap, m_groups_dict): continue
		else: filtered_list.append(inmap_fl)
	return filtered_list


## functions: permutation & inversion rule
def perm_inmap_fl(inmap_fl, perm):
	dim_m = len(inmap_fl)//2

	# reshape inmap to matrix form
	inmap = np.reshape(inmap_fl, [dim_m, 2])
	inmap_mat = np.zeros([dim_m, 2, dim_m])
	inmap_mat[np.arange(dim_m), 0, inmap[:,0]] = 1
	inmap_mat[np.arange(dim_m), 1, inmap[:,1]] = 1

	# permute inmap_mat & convert them back to inmap_fl
	inmap_mat_perm = inmap_mat[perm,:,:][:,:,perm]
	idx_perm = np.where(inmap_mat_perm==1)
	inmap_perm = inmap.copy()
	inmap_perm[idx_perm[0],idx_perm[1]] = idx_perm[2] # back to inmap
	inmap_fl_perm = tuple(inmap_perm.flatten()) # back to inmap_fl
	return inmap_fl_perm

def filter_repeats(inmap_fl_list, outmap):
	'''
	iter thru inmap:
	1. check if i-th inmap has a zero count so far, skip if it doesn't
	2. find outmap-invariant permuted version for each inmap
	3. add 1 to repeat_count for these permuted.
	4. keep those inmaps with repeat_count=0
	'''
	# generate inv_perm_list
	inv_perm_list = gen_inv_perm_list(outmap)

	# hash all inmaps into dict
	repeat_count_dict = {}
	for inmap_fl in inmap_fl_list: repeat_count_dict[inmap_fl] = 0

	# permute i-th inmap
	for i, inmap_fl in enumerate(inmap_fl_list):
		# if i % max(len(inmap_fl_list)//10,1)==0: print(i)

		# check if current repeat_count_dict[inmap_fl] = 0
		if repeat_count_dict[inmap_fl]>0: continue

		# permute inmaps
		inmap_perms = []
		for perm in inv_perm_list:
			inmap_fl_perm = perm_inmap_fl(inmap_fl, perm)
			try:
				if inmap_fl_perm!=inmap_fl: repeat_count_dict[inmap_fl_perm] += 1
			except KeyError: pass

	# append nonrepeated
	filtered_list = []
	for i, inmap_fl in enumerate(repeat_count_dict):
		if repeat_count_dict[inmap_fl]==0: filtered_list.append(inmap_fl)
	return filtered_list

def filter_repeats_with_linmap(inmap_fl_list, linmap_list, outmap):
	'''
	iter thru inmap:
	1. check if i-th inmap has a zero count so far, skip if it doesn't
	2. find outmap-invariant permuted version for each inmap
	3. add 1 to repeat_count for these permuted.
	4. keep those inmaps with repeat_count=0
	'''
	# generate inv_perm_list
	inv_perm_list = gen_inv_perm_list(outmap)

	# hash all inmaps into dict
	repeat_count_dict = {}
	for inmap_fl in inmap_fl_list: repeat_count_dict[inmap_fl] = 0

	# permute i-th inmap
	for i, (inmap_fl, linmap_fl) in enumerate(zip(inmap_fl_list, linmap_list)):
		# if i % max(len(inmap_fl_list)//10,1)==0: print(i)

		# check if current repeat_count_dict[inmap_fl] = 0
		if repeat_count_dict[inmap_fl]>0: continue

		# permute inmaps
		inmap_perms = []
		for perm in inv_perm_list:
			inmap_fl_perm = perm_inmap_fl(inmap_fl, perm)
			try:
				if inmap_fl_perm!=inmap_fl: repeat_count_dict[inmap_fl_perm] += 1
			except KeyError: pass

	# append nonrepeated
	filtered_inmap_fl_list = []
	filtered_linmap_list = []
	for i, (inmap_fl, linmap_fl) in enumerate(zip(inmap_fl_list, linmap_list)):
		if repeat_count_dict[inmap_fl]==0:
			filtered_inmap_fl_list.append(inmap_fl)
			filtered_linmap_list.append(linmap_fl)
	#
	# for i, inmap_fl in enumerate(repeat_count_dict):
	# 	if repeat_count_dict[inmap_fl]==0: filtered_list.append(inmap_fl)
	return filtered_inmap_fl_list, filtered_linmap_list

## functions: irreducible & aperiodic rule
def filter_sinks_and_drains(inmap_fl_list, outmap):
	'''
	1. no 1st-order sinks: no double self-transitions
	2. no 1st-order drain: all the non-self transitions covers all states
	'''
	filtered_list = []
	for i, inmap_fl in enumerate(inmap_fl_list):
		# if i % max(len(inmap_fl_list)//10,1)==0: print(i)
		inmap = np.reshape(inmap_fl, [len(outmap), 2])

		# check sinks
		num_sinks = sum([tuple(x)==(i,i) for i,x in enumerate(inmap)])

		# check drains
		inmap0 = np.array([np.arange(len(outmap)),np.arange(len(outmap))]).T
		num_used_states = np.unique(inmap[np.where(inmap!=inmap0)]).size

		# append
		if (num_sinks==0) & (num_used_states==len(outmap)): filtered_list.append(inmap_fl)
	return filtered_list

def filter_reducible_and_periodic(inmap_fl_list, outmap):
	'''
	convert inmap to Markov chain & keep if it is irreducible & aperiodic
	'''
	filtered_list = []
	for i, inmap_fl in enumerate(inmap_fl_list):
		# if i % max(len(inmap_fl_list)//10,1)==0: print('processing', i)
		inmap = np.reshape(inmap_fl, [len(outmap), 2])
		inmap_mat = np.zeros([2, len(inmap), len(inmap)])
		inmap_mat[0, np.arange(len(inmap)), inmap[:,0]] = .4
		inmap_mat[1, np.arange(len(inmap)), inmap[:,1]] = .6
		inmap_markov = inmap_mat.sum(0)

		# check irr & aper
		mc = qe.MarkovChain(inmap_markov)
		if mc.is_irreducible:
			if mc.is_aperiodic:
				filtered_list.append(inmap_fl)
	return filtered_list


## functions: combined filters
def gen_inmap_fl_list_with_filters(outmap, m_groups_dict):
	print('outmap1 =', outmap)
	# enumerate tabular inmaps
	states = set(np.arange(len(outmap)))
	inmap_fl_list = [x for x in product(states, repeat=len(outmap)*2)]

	# apply fast filters
	inmap_fl_list = filter_sinks_and_drains(inmap_fl_list, outmap)
	inmap_fl_list = filter_mergers(inmap_fl_list, outmap, m_groups_dict)
	inmap_fl_list = filter_repeats(inmap_fl_list, outmap)

	# apply slow filters
	inmap_fl_list = filter_reducible_and_periodic(inmap_fl_list, outmap)
	return inmap_fl_list

def gen_program_list_from_inmap_fl_list(inmap_fl_list, outmap):
	program_list = []
	for inmap_fl in inmap_fl_list:
		try: inmap = np.reshape(inmap_fl, [len(outmap), 2])
		except ValueError: inmap = np.reshape(inmap_fl[0], [len(outmap), 2])
		program_list.append((outmap, inmap))
	return program_list


## functions: move to endnodes + min wiring
def gen_perm_prog(prog, prog_lin, perm):
	'''
	note: if no linmap, just use inmap twice
	'''
	# load
	outmap, inmap = prog
	loutmap, linmap = prog_lin

	# convert inmap to transition matrix
	inmap_mat = np.zeros([len(outmap), 2, len(outmap)])
	inmap_mat[np.arange(len(outmap)), 0, inmap[:,0]] = 1
	inmap_mat[np.arange(len(outmap)), 1, inmap[:,1]] = 1

	# permute
	outmap_perm = tuple(np.array(outmap)[list(perm)])
	loutmap_perm = tuple(np.array(loutmap)[list(perm)])
	linmap_perm = linmap[list(perm),:]

	inmap_mat_perm = inmap_mat[perm,:,:][:,:,perm]
	idx_perm = np.where(inmap_mat_perm==1)
	#
	inmap_perm = inmap.copy()
	inmap_perm[idx_perm[0],idx_perm[1]] = idx_perm[2] # back to inmap
	#
	return (outmap_perm, inmap_perm), (loutmap_perm, linmap_perm)

def gen_stdz_prog(prog, prog_lin, use_perm=False, stdz_outmap=True, perm=(0,1,2,3,4)):
	'''
	note: if no linmap, just use inmap twice
	1) standardize outmap
	2) move winning endnodes to two ends
	3) find minimal wiring
	'''
	if use_perm:
		return gen_perm_prog(prog, prog_lin, perm)

	# standardize outmap
	if stdz_outmap:
		perm0 = np.argsort(prog[0])
		prog, prog_lin = gen_perm_prog(prog, prog_lin, perm0)
		outmap4perm = prog[0]
	else:
		outmap4perm = (prog[1][:,1]==np.arange(len(prog[0]))) * np.array(prog[0]) #e.g., (-1,0,1,0,0)
		perm0 = np.argsort(outmap4perm)
		outmap4perm = outmap4perm[perm0]
		prog, prog_lin = gen_perm_prog(prog, prog_lin, perm0)

	# all possible perms that satisfy
	perm_list = gen_inv_perm_list(outmap4perm, keep_unpermuted=True, keep_flipped=False)

	# find winning endnode score
	score_prog_dict = {}
	for perm in perm_list:
		prog1, prog1_lin = gen_perm_prog(prog, prog_lin, perm)
		outmap1, inmap1 = prog1
		loutmap1, linmap1 = prog1_lin
		#
		endnode_score = ((inmap1[:,1]==np.arange(len(outmap1))) * np.cumsum(outmap1)).sum()
		wiring_score = -np.abs(inmap1 - np.arange(len(outmap1))[:,None]).sum()
		#
		score_prog_dict[(endnode_score, wiring_score)] = (prog1, prog1_lin), perm

	# find minimal wiring with endnodes
	(prog, prog_lin), perm = score_prog_dict[sorted(score_prog_dict)[-1]]

	# combined perm
	perm_combined = perm0[perm]

	# find minimal wiring with endnodes
	return prog, prog_lin, perm_combined




# # # #%%
# tarP = ((-1, 1, 1, 1, 1),
#  np.array([[1, 0],
# 		[0, 2],
# 		[3, 4],
# 		[1, 3],
# 		[3, 1]]))
# tarP_lin = tarP
#
# tarP, tarP_lin, perm_tar = gen_stdz_prog(tarP, tarP_lin, use_perm=False, stdz_outmap=False)
# perm_tar, tarP
