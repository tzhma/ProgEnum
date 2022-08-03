'''
NOTE:
1) insert of figS_rTE
2) rTE is from df_rpte_268533.csv, df_rpte.gephi
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

if not os.path.exists('fig'): os.makedirs('fig')


## load
extension = '268533'
df_rpte = pd.read_pickle('data/df_rpte_%s'%extension)

numP_gpn = [len(x) for x in df_rpte[df_rpte['if_GPN']==1]['GPN_prog_ids'].values]
numP_conn = [len(x) for x in df_rpte[df_rpte['if_GPN_conn']==1]['GPN_conn_prog_ids'].values]
numP_notgpn = df_rpte[df_rpte['if_GPN']==0]['num_prog'].values


## PLOT
plt.figure(figsize=(4,2), dpi=300)
plt.boxplot([numP_gpn, numP_conn, numP_notgpn], showfliers=False)
plt.xticks([1,2,3], labels=['good prog', 'conn prog', 'other prog'])
plt.savefig('fig/figS_rTE_nodestat.svg')
