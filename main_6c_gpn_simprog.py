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


if not os.path.exists('data'): os.makedirs('data')
if not os.path.exists('plot'): os.makedirs('plot')


## LOAD
pprog_arr = pickle.load(open('data/df_gpn_pprog_arr', 'rb'))


## FUNCTIONS
def gen_simprog_arr():
	'''
	1m
	'''
	pp1 = np.zeros_like(pprog_arr)
	for t in range(pprog_arr.shape[-1]):
		for i in range(len(pprog_arr)):
			pp1[i, np.argsort(pprog_arr[i,:,t])[::-1][:int(np.around(1/pprog_arr[i,i,t]))], t] = 1
	#
	pp2 = pp1#((pp1 + np.transpose(pp1, [1,0,2]))>0)*1
	pp3 = pp2.sum(-1)
	return pp2, pp3


## RUN
if False:
	pp2, pp3 = gen_simprog_arr()

	# pickle
	# pickle.dump(pp2, open('data/df_gpn_simprog_arr_list', 'wb'))
	pickle.dump(pp3, open('data/df_gpn_simprog_arr', 'wb'))


##%% PLOT: 10 hists
plt.figure(dpi=300)
for t in range(1,11):
	plt.hist(pp2[:,:,:t].sum(-1).flatten(), bins=np.arange(12)-.5, rwidth=.8, label='ao-seq-depth=%s'%t)
	plt.xlabel('program pair similarity')
	plt.legend()
	plt.savefig('plot/simprog_%s.png'%t)
