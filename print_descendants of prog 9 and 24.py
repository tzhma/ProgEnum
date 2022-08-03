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


## LOAD
df_pte = pd.read_pickle('data/df_gpn_pte')
df_arr = pd.read_pickle('data/df_gpn_seq_motif_0')


##%% find another descendant of prog 9
tar_9 = [9] + df_pte[df_pte['source']==9].index.tolist()
dff = df_arr[df_arr['level']=='global']
{x:dff[dff['prog_id']==x]['motifs'].values[0] for x in tar_9}


##%% find another descendant of prog 24
tar_24 = [24] + df_pte[df_pte['source']==24].index.tolist()
dff = df_arr[df_arr['level']=='global']
{x:dff[dff['prog_id']==x]['motifs'].values[0] for x in tar_24}
