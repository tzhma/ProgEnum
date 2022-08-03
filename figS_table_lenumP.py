'''
NOTE:
1) insert of figS_lenumP
2) evolutionary tree is from df_lenumP5.csv, df_lenumP.gephi
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

## LOAD
df = pd.read_pickle('data/df_enumP_para14')
df_lenumP5 = pd.read_pickle('data/df_lenumP5')
df_lenumP4 = pd.read_pickle('data/df_lenumP4')


##%% TABLE, M<=5
counts_full = [
len(df),
len(df[df['eR']>=.277911]),
len(df[(df['eR']<.277911) & (df['eR']>=.25)]),
len(df[df['eR']<.25])
]
counts = [
len(df_lenumP5),
len(df_lenumP5[df_lenumP5['eR']>=.277911]),
len(df_lenumP5[(df_lenumP5['eR']<.277911) & (df_lenumP5['eR']>=.25)]),
len(df_lenumP5[df_lenumP5['eR']<.25])
]
print(counts_full)
print(counts)

##%% TABLE, M<=4
counts_full = [
len(df[df['progsize']<=4]),
len(df[(df['progsize']<=4) & (df['eR']>=.277911)]),
len(df[(df['progsize']<=4) & (df['eR']<.277911) & (df['eR']>=.25)]),
len(df[(df['progsize']<=4) & (df['eR']<.25)])
]
counts = [
len(df_lenumP4),
len(df_lenumP4[df_lenumP4['eR']>=.277911]),
len(df_lenumP4[(df_lenumP4['eR']<.277911) & (df_lenumP4['eR']>=.25)]),
len(df_lenumP4[df_lenumP4['eR']<.25])
]
print(counts_full)
print(counts)
