'''
single: 2m to generate mergP_dict
BE AWARE OF CHOOSING RIGHT mergP_dict
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


## functions: edit-distance
# def gen_perm_prog_dict(program_list):
# 	'''
# 	what for: gen_edit_distance_list
# 	note: This generates full list of permuted programs (not just outmap invariant ones) for it is used to compute edit distance between any two programs (including changing action assigned to a node)
# 	'''
# 	perm_prog_dict = {}
# 	for i, (outmap, inmap) in enumerate(program_list):
# 		try: inmap_fl = tuple(inmap.flatten())
# 		except AttributeError:
# 			inmap_fl = inmap
# 			inmap = np.reshape(inmap, [len(outmap), 2])
# 		key = (outmap, inmap_fl)
#
# 		# convert inmap to transition matrix
# 		inmap_mat = np.zeros([len(outmap), 2, len(outmap)])
# 		inmap_mat[np.arange(len(outmap)), 0, inmap[:,0]] = 1
# 		inmap_mat[np.arange(len(outmap)), 1, inmap[:,1]] = 1
#
# 		# permute outmaps
# 		perm_list = list(permutations(range(len(outmap))))
# 		outmap_perms = [tuple(np.array(outmap)[list(perm)]) for perm in perm_list]
#
# 		# permute inmap_mat & convert them back to inmap_fl
# 		inmap_mat_perms = [inmap_mat[perm,:,:][:,:,perm] for perm in perm_list]
# 		prog_fl_perms = []
# 		for outmap_perm, inmap_mat_perm in zip(outmap_perms, inmap_mat_perms):
# 			idx_perm = np.where(inmap_mat_perm==1)
# 			inmap_perm = inmap.copy()
# 			inmap_perm[idx_perm[0],idx_perm[1]] = idx_perm[2] # back to inmap
# 			inmap_fl_perm = tuple(inmap_perm.flatten()) # back to inmap_fl
# 			prog_fl_perm_0 = outmap_perm + inmap_fl_perm
# 			prog_fl_perm_1 = tuple(-np.array(outmap_perm)) + inmap_fl_perm # flipping outmap makes the same program
# 			prog_fl_perms.append(prog_fl_perm_0)
# 			prog_fl_perms.append(prog_fl_perm_1)
#
# 		# add to dict
# 		perm_prog_dict[key] = np.array(prog_fl_perms) # mat.shape = (2*M!, 3*M)
# 	return perm_prog_dict

def gen_perm_prog_array(program, flipP_included=True):
	'''
	all permuted version of program is flatten and put into an array.shape = (2*M!,3*M)
	'''
	outmap, inmap = program
	try: inmap_fl = tuple(inmap.flatten())
	except AttributeError:
		inmap_fl = inmap
		inmap = np.reshape(inmap, [len(outmap), 2])

	# convert inmap to transition matrix
	inmap_mat = np.zeros([len(outmap), 2, len(outmap)])
	inmap_mat[np.arange(len(outmap)), 0, inmap[:,0]] = 1
	inmap_mat[np.arange(len(outmap)), 1, inmap[:,1]] = 1

	# permute outmaps
	perm_list = list(permutations(range(len(outmap))))
	outmap_perms = [tuple(np.array(outmap)[list(perm)]) for perm in perm_list]

	# permute inmap_mat & convert them back to inmap_fl
	inmap_mat_perms = [inmap_mat[perm,:,:][:,:,perm] for perm in perm_list]
	prog_fl_perms = []
	for outmap_perm, inmap_mat_perm in zip(outmap_perms, inmap_mat_perms):
		idx_perm = np.where(inmap_mat_perm==1)
		inmap_perm = inmap.copy()
		inmap_perm[idx_perm[0],idx_perm[1]] = idx_perm[2] # back to inmap
		inmap_fl_perm = tuple(inmap_perm.flatten()) # back to inmap_fl
		prog_fl_perm_0 = outmap_perm + inmap_fl_perm
		prog_fl_perms.append(prog_fl_perm_0)
		#
		if flipP_included:
			prog_fl_perm_1 = tuple(-np.array(outmap_perm)) + inmap_fl_perm # flipping outmap makes the same program
			prog_fl_perms.append(prog_fl_perm_1)
	return np.array(prog_fl_perms)

def gen_edit_distance(perm_prog_arr, program):
	try: prog_fl = tuple(program[0]) + tuple(program[1].flatten())
	except AttributeError: prog_fl = program[0] + program[1]
	edit_distance = ((np.array(prog_fl) - perm_prog_arr)!=0).sum(1).min()
	return edit_distance

# def gen_edit_distance_list(perm_prog_arr, program_list):
# 	'''
# 	1) program is the one to be permuted => perm_prog_arr
# 	2) all programs need to have the same size
# 	3) edit_distance = min_i([outmap_0, inmap_fl_0] - [outmap_1, inmap_fl_1]_perm_i)
# 	4) This generates edit distances between "program" relative to a list of programs: "perm_prog_dict.keys()"
# 	'''
# 	# convert to program_fl
# 	# try: prog_fl_0 = tuple(program[0]) + tuple(program[1].flatten())
# 	# except AttributeError: prog_fl_0 = program[0] + program[1]
# 	try: prog_fl_list = [tuple(x[0]) + tuple(x[1].flatten()) for x in program_list]
# 	except AttributeError: prog_fl_list = [x[0] + x[1] for x in program_list]
#
# 	# compute edit distances
# 	edit_distance_list = []
# 	for prog_fl in prog_fl_list:
# 		edit_distance = ((np.array(prog_fl) - perm_prog_arr)!=0).sum(1).min()
# 		edit_distance_list.append(edit_distance)
# 	return edit_distance_list


def gen_edit_distance_list(perm_prog_arr, program_list):
	'''
	1) program is the one to be permuted => perm_prog_arr
	2) all programs need to have the same size
	3) edit_distance = min_i([outmap_0, inmap_fl_0] - [outmap_1, inmap_fl_1]_perm_i)
	4) This generates edit distances between "program" relative to a list of programs: "perm_prog_dict.keys()"
	'''
	# convert to program_fl
	# try: prog_fl_0 = tuple(program[0]) + tuple(program[1].flatten())
	# except AttributeError: prog_fl_0 = program[0] + program[1]
	try: prog_fl_list = [tuple(x[0]) + tuple(x[1].flatten()) for x in program_list]
	except AttributeError: prog_fl_list = [x[0] + x[1] for x in program_list]

	# cast prog_fl_list into array
	prog_arr = np.array(prog_fl_list) # shape=(num_prog, 3*M)

	# compute edit distances
	d_arr = ((prog_arr[:,None,:] - perm_prog_arr[None,:,:])!=0).sum(-1) # shape=(num_prog, num_perm)
	d_list = d_arr.min(-1).tolist() # shape=(num_prog,)

	# extract perm_prog that minimizes d_arr
	progsize = len(program_list[0][0])
	min_permP_list = [perm_prog_arr[x] for x in d_arr.argmin(-1)]
	min_permP_list = [(tuple(x[:progsize]), tuple(x[progsize:]))  for x in min_permP_list]
	return d_list, min_permP_list


## functions: merger-programs

# def gen_one_merger_prog_list(program, part):
# 	'''
# 	note: need len(outmap)==len(part)
# 	'''
# 	outmap, inmap = program
# 	try: inmap_fl = tuple(inmap.flatten())
# 	except AttributeError: inmap_fl = inmap
#
# 	# permute partition & fill in original states in order
# 	part_perms = sorted(set(permutations(part)), reverse=True)
#
# 	# construct all merger programs under such a partition
# 	prog_fl_list = []
# 	for part_perm in part_perms:
# 		m_added = np.arange(len(outmap), sum(part_perm)) # nodes to be added
# 		m_split = (np.array(part_perm)-1).cumsum()
# 		m_added2groups = [tuple(x) for x in np.split(range(len(outmap), sum(part_perm)), m_split)][:-1]
#
# 		# construct mapping & inverse mapping for all nodes
# 		m1m_dict = {i:(i,)+x for i,x in enumerate(m_added2groups)}
# 		mm1_dict_list = [{x:i for x in y} for i,y in enumerate(m1m_dict.values())]
# 		mm1_dict = {}
# 		for x in mm1_dict_list: mm1_dict.update(x)
#
# 		# permute consistent winning & losing edges for all nodes
# 		inmap_ext = [inmap[mm1_dict[x]] for x in range(sum(part_perm))] # this yields inmap after merger relabling
# 		inmap_fl_ext4perm = fl([[m1m_dict[x] for x in y] for y in inmap_ext])
# 		inmap1_fl_perms = list(product(*inmap_fl_ext4perm))
#
# 		# construct outmap
# 		outmap1 = np.array([outmap[mm1_dict[x]] for x in range(sum(part_perm))])
# 		outmap1 = tuple(outmap1 * ((sum(outmap1)>=0)*2-1)) # flip outmap when more -1s than +1s
#
# 		# filters
# 		inmap1_fl_perms = filter_sinks_and_drains(inmap1_fl_perms, outmap1)
# 		inmap1_fl_perms = filter_reducible_and_periodic(inmap1_fl_perms, outmap1)
#
# 		# standardize by sorting outmap
# 		perm = np.argsort(outmap1)
# 		outmap1_stdz = tuple(np.array(outmap1)[perm])
# 		inmap1_fl_perms_stdz = []
# 		for inmap1_fl_perm in inmap1_fl_perms:
# 			inmap1_fl_stdz = perm_inmap_fl(inmap1_fl_perm, perm)
# 			inmap1_fl_perms_stdz.append(inmap1_fl_stdz)
#
# 		# append
# 		prog_fl_lis = [(outmap1_stdz, x) for x in inmap1_fl_perms_stdz]
# 		prog_fl_list += prog_fl_lis
# 		# print('part_perm = %s: %s programs' % (part_perm, len(prog_fl_lis)))
#
# 	# print('part = %s: %s programs before repeats removal' % (part, len(prog_fl_list)))
#
# 	# group inmaps according their outmaps
# 	outmap_list = list(set([x[0] for x in prog_fl_list]))
# 	inmap_fl_lists = []
# 	for outmap in outmap_list:
# 		inmap_fl_list = [x[1] for x in prog_fl_list if x[0]==outmap]
# 		inmap_fl_lists.append(inmap_fl_list)
#
# 	# final removal of repeats
# 	prog_fl_list = []
# 	for outmap, inmap_fl_list in zip(outmap_list, inmap_fl_lists):
# 		inmap_fl_list = filter_repeats(inmap_fl_list, outmap)
# 		prog_fl_lis = [(outmap, x) for x in inmap_fl_list]
# 		prog_fl_list += prog_fl_lis
#
# 	# print('part = %s: %s programs after repeats removal' % (part, len(prog_fl_list)))
# 	return prog_fl_list
#

def gen_one_merger_prog_list(program, part, flipP_included=True):
	'''
	note: need len(outmap)==len(part)
	'''
	outmap, inmap = program
	try: inmap_fl = tuple(inmap.flatten())
	except AttributeError: inmap_fl = inmap

	# permute partition & fill in original states in order
	part_perms = sorted(set(permutations(part)), reverse=True)

	# construct all merger programs under such a partition
	prog_fl_list = []
	for part_perm in part_perms:
		m_added = np.arange(len(outmap), sum(part_perm)) # nodes to be added
		m_split = (np.array(part_perm)-1).cumsum()
		m_added2groups = [tuple(x) for x in np.split(range(len(outmap), sum(part_perm)), m_split)][:-1]

		# construct mapping & inverse mapping for all nodes
		m1m_dict = {i:(i,)+x for i,x in enumerate(m_added2groups)}
		mm1_dict_list = [{x:i for x in y} for i,y in enumerate(m1m_dict.values())]
		mm1_dict = {}
		for x in mm1_dict_list: mm1_dict.update(x)

		# permute consistent winning & losing edges for all nodes
		inmap_ext = [inmap[mm1_dict[x]] for x in range(sum(part_perm))] # this yields inmap after merger relabling
		inmap_fl_ext4perm = fl([[m1m_dict[x] for x in y] for y in inmap_ext])
		inmap1_fl_perms = list(product(*inmap_fl_ext4perm))

		# construct outmap
		outmap1 = np.array([outmap[mm1_dict[x]] for x in range(sum(part_perm))])
		if flipP_included: outmap1 = tuple(outmap1 * ((sum(outmap1)>=0)*2-1)) # flip outmap when more -1s than +1s

		# filters
		inmap1_fl_perms = filter_sinks_and_drains(inmap1_fl_perms, outmap1)
		inmap1_fl_perms = filter_reducible_and_periodic(inmap1_fl_perms, outmap1)

		# standardize by sorting outmap
		perm = np.argsort(outmap1)
		# linmap = tuple([{perm[x]:y for x,y in mm1_dict.items()}[z] for z in range(sum(part_perm))]) #bug
		linmap = tuple([mm1_dict[z] for z in perm])
		outmap1_stdz = tuple(np.array(outmap1)[perm])
		inmap1_fl_perms_stdz = []
		for inmap1_fl_perm in inmap1_fl_perms:
			inmap1_fl_stdz = perm_inmap_fl(inmap1_fl_perm, perm)
			inmap1_fl_perms_stdz.append(inmap1_fl_stdz)

		# append
		prog_fl_lis = [(outmap1_stdz, x, linmap) for x in inmap1_fl_perms_stdz]
		prog_fl_list += prog_fl_lis
		# print('part_perm = %s: %s programs' % (part_perm, len(prog_fl_lis)))

	# print('part = %s: %s programs before repeats removal' % (part, len(prog_fl_list)))

	# group inmaps according their outmaps
	outmap_list = list(set([x[0] for x in prog_fl_list]))

	inmap_fl_lists = []
	linmap_lists = []
	for outmap in outmap_list:
		inmap_fl_list = [x[1] for x in prog_fl_list if x[0]==outmap]
		inmap_fl_lists.append(inmap_fl_list)
		linmap_list = [x[2] for x in prog_fl_list if x[0]==outmap]
		linmap_lists.append(linmap_list)

	# final removal of repeats
	prog_fl_list = []
	for outmap, inmap_fl_list, linmap_list in zip(outmap_list, inmap_fl_lists, linmap_lists):
		inmap1_fl_list, linmap1_list = filter_repeats_with_linmap(inmap_fl_list, linmap_list, outmap)
		prog_fl_lis = [(outmap, x, y) for x,y in zip(inmap1_fl_list, linmap1_list)]
		prog_fl_list += prog_fl_lis

	# print('part = %s: %s programs after repeats removal' % (part, len(prog_fl_list)))
	return prog_fl_list


#%%
# df_enumP = pd.read_pickle('data/df_enumP_para14')
# df_enumP.loc[35,'program']
#
# program = (np.array([-1,  1,  1]), np.array([[2, 0],[2, 1],[0, 1]]))
# part = (2,1,1)
#
# gen_one_merger_prog_list(program, part)
#
# #%%
#
# outmap, inmap = program
# try: inmap_fl = tuple(inmap.flatten())
# except AttributeError: inmap_fl = inmap
#
# # permute partition & fill in original states in order
# part_perms = sorted(set(permutations(part)), reverse=True)
#
# # construct all merger programs under such a partition
# prog_fl_list = []
#
#
#
# for part_perm in part_perms:
# 	# part_perm = part_perms[1]
# 	m_added = np.arange(len(outmap), sum(part_perm)) # nodes to be added
# 	m_split = (np.array(part_perm)-1).cumsum()
# 	m_added2groups = [tuple(x) for x in np.split(range(len(outmap), sum(part_perm)), m_split)][:-1]
#
# 	# construct mapping & inverse mapping for all nodes
# 	m1m_dict = {i:(i,)+x for i,x in enumerate(m_added2groups)}
# 	mm1_dict_list = [{x:i for x in y} for i,y in enumerate(m1m_dict.values())]
# 	mm1_dict = {}
# 	for x in mm1_dict_list: mm1_dict.update(x)
#
# 	# permute consistent winning & losing edges for all nodes
# 	inmap_ext = [inmap[mm1_dict[x]] for x in range(sum(part_perm))] # this yields inmap after merger relabling
# 	inmap_fl_ext4perm = fl([[m1m_dict[x] for x in y] for y in inmap_ext])
# 	inmap1_fl_perms = list(product(*inmap_fl_ext4perm))
#
# 	# construct outmap
# 	outmap1 = np.array([outmap[mm1_dict[x]] for x in range(sum(part_perm))])
# 	outmap1 = tuple(outmap1 * ((sum(outmap1)>=0)*2-1)) # flip outmap when more -1s than +1s
#
# 	# filters
# 	inmap1_fl_perms = filter_sinks_and_drains(inmap1_fl_perms, outmap1)
# 	inmap1_fl_perms = filter_reducible_and_periodic(inmap1_fl_perms, outmap1)
#
# 	# standardize by sorting outmap
# 	perm = np.argsort(outmap1)
# 	# linmap = tuple([{perm[x]:y for x,y in mm1_dict.items()}[z] for z in range(sum(part_perm))]) #bug
# 	linmap = tuple([mm1_dict[z] for z in perm])
# 	print(perm)
# 	outmap1_stdz = tuple(np.array(outmap1)[perm])
# 	inmap1_fl_perms_stdz = []
# 	for inmap1_fl_perm in inmap1_fl_perms:
# 		inmap1_fl_stdz = perm_inmap_fl(inmap1_fl_perm, perm)
# 		inmap1_fl_perms_stdz.append(inmap1_fl_stdz)
#
# 	# append
# 	prog_fl_lis = [(outmap1_stdz, x, linmap) for x in inmap1_fl_perms_stdz]
# 	prog_fl_list += prog_fl_lis
# 	# print('part_perm = %s: %s programs' % (part_perm, len(prog_fl_lis)))
#
# # print('part = %s: %s programs before repeats removal' % (part, len(prog_fl_list)))
#
# # group inmaps according their outmaps
# outmap_list = list(set([x[0] for x in prog_fl_list]))
#
# inmap_fl_lists = []
# linmap_lists = []
# for outmap in outmap_list:
# 	# linmap = list(set([x[-1] for x in prog_fl_list if x[0]==outmap]))[0]
# 	# linmap_list.append(linmap)
# 	inmap_fl_list = [x[1] for x in prog_fl_list if x[0]==outmap]
# 	inmap_fl_lists.append(inmap_fl_list)
# 	linmap_list = [x[2] for x in prog_fl_list if x[0]==outmap]
# 	linmap_lists.append(linmap_list)
#
# # final removal of repeats
# prog_fl_list = []
# for outmap, inmap_fl_list, linmap_list in zip(outmap_list, inmap_fl_lists, linmap_lists):
# 	inmap1_fl_list, linmap1_list = filter_repeats_with_linmap(inmap_fl_list, linmap_list, outmap)
# 	prog_fl_lis = [(outmap, x, y) for x,y in zip(inmap1_fl_list, linmap1_list)]
# 	prog_fl_list += prog_fl_lis
#
# # print('part = %s: %s programs after repeats removal' % (part, len(prog_fl_list)))
# prog_fl_list



# #%%


# # df_enumP = pd.read_pickle('data/df_enumP_para14')
# program = df_enumP.loc[8, 'program']
# gen_one_merger_prog_list(program, (3,1,1))


def gen_part_list_for_merger_prog(progsize, merger_progsize):
	'''
	this generates partition(merger_progsize) with its length=progsize
	'''
	valid_part_list = [x for x in partition(merger_progsize) if len(x)==progsize]
	return valid_part_list

def gen_mergP_dict(prog_id_list, progsize_max=5, flipP_included=True):
	'''
	pre-generate this for find_prog_with_d1
	'''
	# load with id_unique for each program
	df_enumP = pd.read_pickle('data/df_enumP')

	for prog_id in prog_id_list:
		progsize = df_enumP.loc[prog_id, 'progsize']
		program = df_enumP.loc[prog_id, 'program']

		# generate mergProg
		part_list = fl([gen_part_list_for_merger_prog(progsize,x) for x in range(progsize+1, progsize_max+1)])
		for part in part_list:
			try: mergP_dict[(prog_id, part)]
			except KeyError:
				print(prog_id, part)
				mergP_list = gen_one_merger_prog_list(program, part, flipP_included)
				mergP_dict[(prog_id, part)] = mergP_list
	return mergP_dict

def gen_mergP_dict_gpn():
	'''
	pre-generate this for find_prog_with_d1
	'''
	progsize_max = 5

	# load with id_unique for each program
	df_gpn = pd.read_pickle('data/df_gpn')
	uni_list = df_gpn['id_unique'].values
	progsize_list = df_gpn['progsize'].values
	prog_list = pickle.load(open('data/df_gpn_flip_prog_list','rb'))
	#
	for uni, progsize, program in zip(uni_list, progsize_list, prog_list):
		# progsize = df_enumP.loc[prog_id, 'progsize']
		# program = df_enumP.loc[prog_id, 'program']

		# generate mergProg
		part_list = fl([gen_part_list_for_merger_prog(progsize,x) for x in range(progsize+1, progsize_max+1)])
		for part in part_list:
			try: mergP_dict[(uni, part)]
			except KeyError:
				print(uni, part)
				mergP_list = gen_one_merger_prog_list(program, part, flipP_included=True)
				mergP_dict[(uni, part)] = mergP_list
	return mergP_dict


## functions: program space
def find_prog_with_d1(prog_id, df_enumP, first_prog_id_0=0, flipP_included=True):
	'''
	note: mergP_dict must be globally accessible
	1) given i-th program, check edit-distance (d) from the top until finding the first program (j-th) with d=1
	2) if progsize_i == progsize_j, compute d directly
	3) if progsize_i != progsize_j, going thru the list of partitions for mergProg_j, and stopping when d=1 is found for certain a partition
	4) save those mergProg_j in a global dict for later use: mergP_dict[(prog_id, part)] = gen_one_merger_prog_list(program, part)
	'''
	# load program
	program = df_enumP.loc[prog_id, 'program']

	# permute program outside for-loop!
	perm_prog_arr = gen_perm_prog_array(program)

	# find prog with d=1
	for prog_id_0 in df_enumP.loc[first_prog_id_0:].index:
		if prog_id_0==prog_id: continue

		program_0, id_unique_0 = df_enumP.loc[prog_id_0, ['program', 'id_unique']]
		if len(program_0[0])>len(program[0]):
			return [prog_id, -1, (-1,), [-1]]

		# find mergP that has d=1 to program
		part_list = gen_part_list_for_merger_prog(len(program_0[0]), len(program[0]))
		for part in part_list:
			try: mergP_list = mergP_dict[(id_unique_0, part)]
			except KeyError:
				mergP_list = gen_one_merger_prog_list(program_0, part, flipP_included)
				mergP_dict[(id_unique_0, part)] = mergP_list #if len(mergP_list)>1:
			if len(mergP_list)>0:
				d_list, min_permP_list = gen_edit_distance_list(perm_prog_arr, mergP_list)
			else:
				d_list = [-1]
			if len(d_list)==0: d_list = [-1]

			# append when d=1 is found
			if min(d_list)==1:
				idx_min = np.argmin(d_list)
				# min_permP, min_mergP = min_permP_list[idx_min], mergP_list[idx_min]
				d_hist = [d_list.count(i) for i in range(1, max(d_list)+1)]
				dat_list = [prog_id, prog_id_0, part, d_hist]

				# TEST: 24 -> 7655
				# min_permP, min_mergP = min_permP_list[idx_min], mergP_list[idx_min]
				# dat_list = [prog_id, prog_id_0, part, d_hist, min_mergP]

				return dat_list
	return [prog_id, -1, (-1,), [-1]]


## global variable
try:
	m_groups_dict = pickle.load(open('data/m_groups_dict_enumP','rb'))
except FileNotFoundError:
	print('generating m_groups_dict_enumP...')
	m_groups_dict = gen_m_groups_dict(prog_size_max=5, for_enumDB=True)
	pickle.dump(m_groups_dict, open('data/m_groups_dict_enumP', 'wb'))

try:
	mergP_dict = {}
	mergP_dict = pickle.load(open('data/mergP_dict','rb'))
except FileNotFoundError:
	print('generating mergP_dict...')
	mergP_dict = gen_mergP_dict(range(5108)) # only M<=4 can be mergP
	pickle.dump(mergP_dict, open('data/mergP_dict', 'wb'))


##%% gen mergP without ever flipping a program
if False:
	try:
		mergP_dict = {}
		mergP_dict = pickle.load(open('data/mergP_dict_gpn','rb'))
	except FileNotFoundError:
		print('generating mergP_dict_gpn...')
		mergP_dict = gen_mergP_dict_gpn() # only M<=4 can be mergP
		pickle.dump(mergP_dict, open('data/mergP_dict_gpn', 'wb'))
