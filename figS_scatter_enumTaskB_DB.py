'''
3h on cluster
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
from core_bdpB import run_bdpB


## PARAMETERS
dp, dpm = .3, -.5
h_list = np.around(np.linspace(.05, .32, 28), 6)[:-1].tolist() + np.around(np.linspace(.32, .4, 41), 6).tolist()

idx_coarse = np.arange(27).tolist() + (27+np.arange(41)[::5]).tolist()
idx_fine = np.arange(-41,-15)
h_list_c = np.array(h_list)[idx_coarse]
h_list_f = np.array(h_list)[idx_fine]

## critical h
h_c0 = 1/2 + (1-dpm)/(1+dp**2-dpm**2) - 4/(dp**2+(3-dpm)*(1+dpm))
h_c1 = 0.20317
h_c2 = -dpm/(1-dpm)
h_c3 = (1-dpm)/4 - dp**2/4/(1-dpm)

## LOAD
df = pd.read_pickle('data/df_enumTaskB')
data = pickle.load(open('data/df_enumTaskDB_data','rb'))
data_coarse = data[0], np.array(data[1])[idx_coarse], data[2][:,idx_coarse]
data_fine = data[0], np.array(data[1])[idx_fine], data[2][:,idx_fine]

ratio_pLpl_list = df['ratio_pLpl'].values
pp2_list = df['pp2'].values
pp10_list = df['pp10'].values
eRB_list = df['eRB'].values
eRWSLG_list = df['eRWSLG'].values

rL_list_c = ratio_pLpl_list[idx_coarse]
rL_list_f = ratio_pLpl_list[idx_fine]


##%% FUNCTIONS
def gen_enumTaskDB_data(data):
	'''
	data_coarse = pickle.load(open('data/df_enumTaskDB_data_coarse','rb'))
	data_fine = pickle.load(open('data/df_enumTaskDB_data_fine','rb'))
	'''
	## load
	progsize_list, h_list, eR_arr = data

	## run_bdpB
	eR_B_list = []
	p_arr_B_list = []
	for h in h_list:
		para = h, dp, dpm
		eR_B, p_arr, _ = run_bdpB(para)
		eR_B_list.append(eR_B)
		p_arr_B_list.append(p_arr)

	## extract performance
	performance = np.zeros([len(h_list), len(progsize_list)])-.1
	h_progsize_performance_list = []
	for k, (h, eR_B, eR_ar) in enumerate(zip(h_list, eR_B_list, eR_arr.T)):
		eR_lis = np.around(eR_ar, 6)
		eR_WSLG = eR_lis[0]

		eR_dict = {}
		for progsize, eR in zip(progsize_list, eR_lis):
			try:
				eR0 = eR_dict[progsize]
				if eR>eR0: eR_dict[progsize] = eR
			except KeyError:
				eR_dict[progsize] = eR

		progsize_pops = []
		for i, (progsize, eR) in enumerate(eR_dict.items()):
			if i==0: last_progsize = progsize
			else:
				if eR<=eR_dict[last_progsize]: progsize_pops.append(progsize)
				else: last_progsize = progsize
		for x in progsize_pops: eR_dict.pop(x)

		for progsize, eR in eR_dict.items():
			performance[k, progsize] = (eR-eR_WSLG)/(eR_B-eR_WSLG)
			h_progsize_performance_list.append((np.array(h_list)[k], progsize, (eR-eR_WSLG)/(eR_B-eR_WSLG)))

	return h_progsize_performance_list, performance, h_list


## RUN (10s)
h_progsize_performance_list_c, _, h_list_c = gen_enumTaskDB_data(data_coarse)
h_progsize_performance_list_f, _, h_list_f = gen_enumTaskDB_data(data_fine)


##%%: PLOT: pB vs pWSLG
pB_W = np.array([x['W'] for x in df['pB_WLl']])
pB_l = np.array([x['l'] for x in df['pB_WLl']])
pB_L = np.array([np.around(1-x-y,6) for x,y in zip(pB_W, pB_l)])
pWSLG_W = np.array([x['W'] for x in df['pWSLG_WLl']])
pWSLG_l = np.array([x['l'] for x in df['pWSLG_WLl']])
pp2_list = pB_W*pWSLG_W/(pB_W+pWSLG_W) + pB_l*pWSLG_l/(pB_l+pWSLG_l)

plt.figure(figsize=(10,8), dpi=300)
plt.subplot(211)
plt.plot(h_list, pWSLG_W, label='pWSLG(W)', color='k', linestyle='--')
plt.plot(h_list, pWSLG_l, label='pWSLG(l)', color='gray', linestyle='--')
plt.plot(h_list, pB_W, label='pB(W)', color='k')
plt.plot(h_list, pB_l, label='pB(l)', color='gray')
plt.plot(h_list, pB_L, label='pB(L)', color='tab:cyan')
plt.axvline(x=h_c0, color='deeppink', linestyle='--', linewidth=1)
plt.axvline(x=h_c1, color='deeppink', linestyle='--', linewidth=1)
plt.axvline(x=h_c2, color='deeppink', linestyle='--', linewidth=1)
plt.axvline(x=h_c3, color='deeppink', linestyle='--', linewidth=1)
plt.xlabel('h')
plt.ylabel('p')
plt.legend()

plt.subplot(212)
plt.plot(h_list, pp2_list, 'k--', label='seq_len=2')
plt.plot(h_list, pp10_list, 'k', label='seq_len=10')
plt.axvline(x=h_c0, color='deeppink', linestyle='--', linewidth=1)
plt.axvline(x=h_c1, color='deeppink', linestyle='--', linewidth=1)
plt.axvline(x=h_c2, color='deeppink', linestyle='--', linewidth=1)
plt.axvline(x=h_c3, color='deeppink', linestyle='--', linewidth=1)
plt.xlabel('h')
plt.ylabel('p(WSLG | B)')
plt.legend()

plt.savefig('fig/taskSloppiness_enumTaskB.svg')


##%% PLOT: analytical curves of belief update
if not os.path.exists('fig/belief update curves'): os.makedirs('fig/belief update curves')

def u1(u,a,o, para):
	h, dp, dpm = para
	u1 = (1-2*h) * (a*o*dp + (1+o*dpm)*u) / (a*o*dp*u + (1+o*dpm))
	return u1

# control parameter
h_id = 4 ###

# special h
h0, hc0, hc1, hc2, hc3 = .05, .112, .20317, .333333, .36 #.111801
h = (h0, hc0, hc1, hc2, hc3)[h_id]
para = h, dp, dpm

# special u
u_cr = dp/(1-dpm)
u_gw = (1-2*h) * dp/(1+dpm)
u_gl = (1-2*h) * dp/(1-dpm)
u_ub = ((1-2*h) + ((1+dpm)/dp*h)**2)**.5 - (1+dpm)/dp*h
u_ll = (1-2*h) * (-dp+(1-dpm)*u_ub) / (-dp*u_ub+(1-dpm))

# for plots
u_list = np.linspace(-1,1,200)
un_list = np.linspace(-1,0,200)
up_list = np.linspace(0,1,200)
u_vec = [u_cr, u_gw, u_gl, u_ub, u_ll]
c_vec = ['tab:cyan', 'tab:olive', 'tab:brown', 'tab:pink', 'tab:purple']
size_dict = {
h0: [100,100,100,100,100],
hc0: [100,300,100,100,100],
hc1: [300,100,100,100,100],
hc2: [300,100,100,100,100],
hc3: [300,100,100,100,100]}

# plot all four curves
if False:
	h = .05
	plt.figure(figsize=(4.5,4.5), dpi=300)
	plt.plot(u_list, u1(u_list,-1,-1, para), 'k--', label='a,o = -1,-1')
	plt.plot(u_list, u1(u_list,-1,1, para), 'k', label='a,o = -1,1')
	plt.plot(u_list, u1(u_list,1,-1, para), 'k--', label='a,o = 1,-1')
	plt.plot(u_list, u1(u_list,1,1, para), 'k', label='a,o = 1,1')
	plt.grid()
	plt.xlim([-1,1])
	plt.xticks([-1,0,1])
	plt.yticks([-1,0,1])
	plt.savefig('fig/belief update curves/u1_h=%s.svg'%h)

# plot rewardseeking curves
if True:
	plt.figure(figsize=(4.5,4.5), dpi=300)
	plt.axvline(x=0, color='lightgray', linewidth=1, zorder=1)
	plt.axhline(y=0, color='lightgray', linewidth=1, zorder=1)
	plt.plot(u_list, u_list, color='lightgray', label='a,o = 1,1', linewidth=1, zorder=1)
	plt.plot(un_list, u1(un_list,-1,-1, para), 'k--', label='a,o = -1,-1', zorder=1)
	plt.plot(un_list, u1(un_list,-1,1, para), 'k', label='a,o = -1,1', zorder=1)
	plt.plot(up_list, u1(up_list,1,-1, para), 'k--', label='a,o = 1,-1', zorder=1)
	plt.plot(up_list, u1(up_list,1,1, para), 'k', label='a,o = 1,1', zorder=1)

	# five special u
	plt.scatter(u_vec, u_vec, c=c_vec, s=size_dict[h], edgecolor='w', zorder=2)
	plt.plot([u_ub,u_ub], [u_ll,u_ub], color='lightgray', linestyle='--', linewidth=1, zorder=1)
	plt.plot([u_ll,u_ub], [u_ll,u_ll], color='lightgray', linestyle='--', linewidth=1, zorder=1)
	plt.plot([0,u_gw], [u_gw,u_gw], color='lightgray', linestyle='--', linewidth=1, zorder=1)
	plt.plot([0,u_gl], [u_gl,u_gl], color='lightgray', linestyle='--', linewidth=1, zorder=1)
	plt.plot([u_cr,u_cr], [0,u_cr], color='lightgray', linestyle='--', linewidth=1, zorder=1)

	# shaded regions for forbidden u
	plt.fill_betweenx([-1,1],u_ub,1, color='whitesmoke', zorder=0)
	plt.fill_betweenx([-1,1],-u_ub,-1, color='whitesmoke', zorder=0)
	plt.fill_between([-1,1],u_ub,1, color='whitesmoke', zorder=0)
	plt.fill_between([-1,1],-u_ub,-1, color='whitesmoke', zorder=0)
	if u_ll<u_gw:
		u_inner = max(u_gl,u_ll)
		plt.fill_betweenx([-1,1],u_inner,u_gw, color='whitesmoke', zorder=0)
		plt.fill_betweenx([-1,1],-u_inner,-u_gw, color='whitesmoke', zorder=0)
		plt.fill_between([-1,1],u_inner,u_gw, color='whitesmoke', zorder=0)
		plt.fill_between([-1,1],-u_inner,-u_gw, color='whitesmoke', zorder=0)

	# range
	plt.xlim([-u_ub*1.1507440464564187,u_ub*1.1507440464564187])
	plt.ylim([-u_ub*1.1507440464564187,u_ub*1.1507440464564187])
	plt.xticks([-u_ub,0,u_ub])
	plt.yticks([-u_ub,0,u_ub])
	plt.savefig('fig/belief update curves/u1_rs_h=%s.svg'%h)


##%% PLOT: eR_B
plt.figure(figsize=(10,8), dpi=300)
plt.subplot(211)
plt.plot(h_list, eRB_list, 'k', label='eRB')
plt.plot(h_list, eRWSLG_list, 'k--', label='eRWSLG')
plt.axvline(x=h_c0, color='deeppink', linestyle='--', linewidth=1)
plt.axvline(x=h_c1, color='deeppink', linestyle='--', linewidth=1)
plt.axvline(x=h_c2, color='deeppink', linestyle='--', linewidth=1)
plt.axvline(x=h_c3, color='deeppink', linestyle='--', linewidth=1)
plt.xlabel('h')
plt.ylabel('reward rate')
plt.legend()

plt.subplot(212)
# plt.plot(h_list, eRB_list-eRWSLG_list, 'k', label='eRB-eRWSLG')
plt.semilogy(h_list, (eRB_list-eRWSLG_list) * (np.array(h_list) <= h_c3), 'k', label='eRB-eRWSLG')
plt.axvline(x=h_c0, color='deeppink', linestyle='--', linewidth=1)
plt.axvline(x=h_c1, color='deeppink', linestyle='--', linewidth=1)
plt.axvline(x=h_c2, color='deeppink', linestyle='--', linewidth=1)
plt.axvline(x=h_c3, color='deeppink', linestyle='--', linewidth=1)
plt.xlabel('h')
plt.ylabel('reward rate difference')
plt.legend()

plt.savefig('fig/taskSloppiness_enumTaskB_eR.svg')


##%% PLOT: pB(u)
if not os.path.exists('fig/pu_B_param h'): os.makedirs('fig/pu_B_param h')
# df = pd.read_pickle('data/df_enumTaskB')
df = pd.read_pickle('data/df_enumTaskB_special_h')

# for scatter
c_vec = ['tab:cyan', 'tab:olive', 'tab:brown', 'tab:pink', 'tab:purple']
size_dict = {
h0: [100,100,100,100,100],
hc0: [100,300,100,100,100],
hc1: [300,100,100,100,100],
hc2: [300,100,100,100,100],
hc3: [300,100,100,100,100]}

# plot
h_list = df['h'].values
for h in h_list:
	u_cr = dp/(1-dpm)
	u_gw = (1-2*h) * dp/(1+dpm)
	u_gl = (1-2*h) * dp/(1-dpm)
	u_ub = ((1-2*h) + ((1+dpm)/dp*h)**2)**.5 - (1+dpm)/dp*h
	u_ll = (1-2*h) * (-dp+(1-dpm)*u_ub) / (-dp*u_ub+(1-dpm))
	u_vec = [u_cr, u_gw, u_gl, u_ub, u_ll]
	try: s_vec = size_dict[h]
	except KeyError: s_vec = [100,100,100,100,100]

	plt.figure(figsize=(6,1.5), dpi=300)
	uB = df[df['h']==h]['u_list'].values[0]
	pB = df[df['h']==h]['pB_arr'].values[0]
	idx_star_0 = np.where(np.array(uB)<-.2)[0][-1]+1
	idx_star_1 = np.where(np.array(uB)>.2)[0][0]

	plt.plot(uB, pB, color='k', linewidth=1, zorder=1)
	plt.fill_between(uB[:idx_star_0], pB[:idx_star_0], color='tab:cyan', alpha=1, zorder=0)
	plt.fill_between(uB[idx_star_1:], pB[idx_star_1:], color='tab:cyan', alpha=1, zorder=0)

	plt.axvline(u_cr, color='tab:cyan', linestyle='--', linewidth=1)
	plt.axvline(u_gw, color='tab:olive', linestyle='--', linewidth=1)
	plt.axvline(u_gl, color='tab:brown', linestyle='--', linewidth=1)
	plt.axvline(u_ub, color='tab:pink', linestyle='--', linewidth=1)
	plt.axvline(u_ll, color='tab:purple', linestyle='--', linewidth=1)
	plt.scatter(u_vec, [pB.max()*1.1 for x in u_vec], c=c_vec, s=s_vec, edgecolor='w', zorder=2)

	plt.yticks([])
	plt.savefig('fig/pu_B_param h/pu_B_h=%s.svg'%h)


##%% PLOT: DB
alpha = 0.46 #0.07652169*6

plt.figure(figsize=(10,8), dpi=300)
plt.subplot(211)
xc = [x[0] for x in h_progsize_performance_list_c]
yc = [x[1] for x in h_progsize_performance_list_c]
sc = [x[2]*20+1 for x in h_progsize_performance_list_c]
plt.scatter(xc, yc, s=sc, c='k', label='DB')
plt.plot(h_list_c, 1/np.array(rL_list_c) * alpha, 'k--')
plt.axvline(x=h_c0, color='deeppink', linestyle='--', linewidth=1)
plt.axvline(x=h_c1, color='deeppink', linestyle='--', linewidth=1)
plt.axvline(x=h_c2, color='deeppink', linestyle='--', linewidth=1)
plt.axvline(x=h_c3, color='deeppink', linestyle='--', linewidth=1)
plt.ylim([0,30])
plt.legend()
plt.xlabel('h')
plt.ylabel('M')

plt.subplot(212)
xf = [x[0] for x in h_progsize_performance_list_f]
yf = [x[1] for x in h_progsize_performance_list_f]
sf = [x[2]*80+1 for x in h_progsize_performance_list_f]
plt.scatter(xf, yf, s=sf, c='k', label='DB')
plt.plot(h_list_f, 1/np.array(rL_list_f) * alpha, 'k--', label='pl/pL * alpha from full-B')
plt.axvline(x=h_c2, color='deeppink', linestyle='--', linewidth=1)
plt.axvline(x=h_c3, color='deeppink', linestyle='--', linewidth=1)
plt.ylim([0,15])
plt.yticks(np.arange(0,15,2))
plt.legend()
plt.xlabel('h')
plt.ylabel('M')

plt.savefig('fig/taskSloppiness_enumTaskDB.svg')


#%% PRINT: sanity check if integral of pu agrees with results from sdpB
o,a = -1,1
pu = df.loc[42, 'pB_arr']
uL = [x for x in df.loc[42, 'u_list'] if x>.2]
puL = [y for x,y in zip(df.loc[42, 'u_list'], df.loc[42, 'pB_arr']) if x>.2]
print('pB(L)', sum([(1+o*(a*dp*u+dpm))/2 * pu * 2 for u,pu in zip(uL, puL)]), df.loc[42, 'pB_WLl']['L'])

o,a = -1,1
pu = df.loc[42, 'pB_arr']
ul = [x for x in df.loc[42, 'u_list'] if (x<.2)&(x>0)]
pul = [y for x,y in zip(df.loc[42, 'u_list'], df.loc[42, 'pB_arr']) if (x<.2)&(x>0)]
print('pB(l)', sum([(1+o*(a*dp*u+dpm))/2 * pu * 2 for u,pu in zip(ul, pul)]), df.loc[42, 'pB_WLl']['l'])

#%%
# df.loc[50, 'pWSLG_WLl']
