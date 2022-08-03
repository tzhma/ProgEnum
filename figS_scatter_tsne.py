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


## LOAD
df = pd.read_pickle('data/df_enumP_para14_tsne')
tsne_x, tsne_y = df['tsne_x'].values, df['tsne_y'].values

iDB = df[df['eR']==df['eR'].max()].index[0]
tsne_xDB, tsne_yDB = df.loc[iDB, ['tsne_x', 'tsne_y']].values
tsne_xWSLG, tsne_yWSLG = df.loc[4, ['tsne_x', 'tsne_y']].values

eR_list = df['eR'].values
eR_max = eR_list.max()
eR_min = .25 - (eR_list.max()-.25)

xWSLG = df.loc[4, 'struc_mat']
xDB = df.loc[iDB, 'struc_mat']
d_list = [np.abs(x-xDB).sum() for x in df['struc_mat'].values]


##%% PLOT
np.random.seed(42)

plt.figure(figsize=(12,3.5), dpi=600)
plt.subplot(131)
plt.scatter(tsne_x, tsne_y, cmap='RdGy_r', c=eR_list, s=1.5, alpha=1, edgecolors='none', vmin=eR_min, vmax=eR_max)
plt.colorbar()
plt.xticks([])
plt.yticks([])
plt.scatter(tsne_xDB, tsne_yDB, s=50, marker='o', facecolors='none', edgecolors='deeppink', linewidth=.75)
plt.scatter(tsne_xWSLG, tsne_yWSLG, s=50, marker='o', facecolors='none', edgecolors='deeppink', linewidth=.75)

plt.subplot(132)
plt.scatter(tsne_x, tsne_y, cmap='Spectral_r', c=d_list, s=1.5, alpha=1, edgecolors='none')
plt.colorbar()
plt.xticks([])
plt.yticks([])
plt.scatter(tsne_xDB, tsne_yDB, s=50, marker='o', facecolors='none', edgecolors='deeppink', linewidth=.75)
plt.scatter(tsne_xWSLG, tsne_yWSLG, s=50, marker='o', facecolors='none', edgecolors='deeppink', linewidth=.75)

plt.subplot(133)
d_plot = np.array(d_list) + np.random.randn(len(d_list))*.2
plt.scatter(d_plot, eR_list, c='k', s=2, alpha=.1, edgecolors='none')
plt.scatter([d_plot[4], d_plot[iDB]], [eR_list[4], eR_list[iDB]], s=50, marker='o', facecolors='none', edgecolors='deeppink', linewidth=.75)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('fig/tsne.png')
