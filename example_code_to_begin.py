
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


## LOAD
df_enumP = pd.read_pickle('data/df_enumP_para14_sorted')
df_TE = pd.read_pickle('data/df_gpn')
df_fTE = pd.read_pickle('data/df_gpn_pte')


## PRINT
print(df_enumP.columns)
print(df_TE.columns)
print(df_fTE.columns)


## PLOT
plt.figure()
plt.hist(df_enumP['eR'], bins=100)
plt.xlabel('reward rate')
