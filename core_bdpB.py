'''
NOTE:
BDP is equivalent to computing top eigenvector of joint world MDP & policy from small program
'''
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import os
import sys
from itertools import permutations, combinations, product, islice
from numpy import linalg as LA


## bdp parameters
err_thres = 1e-6
max_iter = 1000
dim_a, dim_o = 2, 2
a_list, o_list = [-1,1], [-1,1]
# dim_u = 200
# u_list = [-1 + 2*i/(dim_u-1) for i in range(dim_u)]
# outmap = np.array([np.array(u_list)<0, np.array(u_list)>0]).T # rewardseek


## functions
def gen_p_dict(u_list, para, dim_u):
	h,dp,dpm = para
	u_ub = ((1-2*h) + ((1+dpm)/dp*h)**2)**.5 - (1+dpm)/dp*h
	u_list = [-u_ub + 2*u_ub*i/(dim_u-2-1) for i in range(dim_u-2)]
	u0, u1 = u_list[0] - np.diff(u_list).mean(), u_list[-1] + np.diff(u_list).mean()
	u_list = [u0] + u_list + [u1]
	y = {}
	for i in range(dim_u):
		du = u_list[i]
		for a, o in list(product(a_list, o_list)):
			key = ((i), (a,o))
			p = (1 + o*(a*dp*du + dpm)) / 2
			if (p>1) or (p<0): y[key] = 0
			else: y[key] = p
	return y

def gen_eR(p_array,p_dict,dim_u,para):
	h,dp,dpm = para
	u_ub = ((1-2*h) + ((1+dpm)/dp*h)**2)**.5 - (1+dpm)/dp*h
	u_list = [-u_ub + 2*u_ub*i/(dim_u-2-1) for i in range(dim_u-2)]
	u0, u1 = u_list[0] - np.diff(u_list).mean(), u_list[-1] + np.diff(u_list).mean()
	u_list = [u0] + u_list + [u1]
	outmap = np.array([np.array(u_list)<0, np.array(u_list)>0]).T # rewardseek
	p_uo_list = []
	for i in range(dim_u):
		pu, du = p_array[i], u_list[i]
		for a in a_list:
			a_id = a_list.index(a)
			key = ((i), (a,1))
			p_oa = p_dict[key] * outmap[i,a_id]
			p_uo_list.append(pu * p_oa)
	return np.sum(p_uo_list)

def updates(i,a,o, para, dim_u):
	h,dp,dpm = para
	u_ub = ((1-2*h) + ((1+dpm)/dp*h)**2)**.5 - (1+dpm)/dp*h
	u_list = [-u_ub + 2*u_ub*i/(dim_u-2-1) for i in range(dim_u-2)]
	u0, u1 = u_list[0] - np.diff(u_list).mean(), u_list[-1] + np.diff(u_list).mean()
	u_list = [u0] + u_list + [u1]
	a_id, o_id = a_list.index(a), o_list.index(o)
	du = u_list[i]
	du1 = (1-2*h) * (a*o*dp + (1+o*dpm)*du) / (a*o*dp*du + 1+o*dpm + 1e-12)
	i1, i2 = np.argsort(np.abs(du1 - np.array(u_list)))[:2]
	p = (u_list[i2] - du1) / (u_list[i2] - u_list[i1])
	return (i1,i2,p)

def ToStationaryDistribution(p_array,outmap,p_dict,para,dim_u,err_thres=err_thres):
	h,dp,dpm = para
	u_ub = ((1-2*h) + ((1+dpm)/dp*h)**2)**.5 - (1+dpm)/dp*h
	u_list = [-u_ub + 2*u_ub*i/(dim_u-2-1) for i in range(dim_u-2)]
	u0, u1 = u_list[0] - np.diff(u_list).mean(), u_list[-1] + np.diff(u_list).mean()
	u_list = [u0] + u_list + [u1]
	y0 = p_array.copy()
	t,err,eR = 0,1,1
	while (err>err_thres) & (t<max_iter):
		y1 = np.zeros(dim_u)
		for i in range(dim_u):
			if y0[i] < .01/dim_u: continue
			for a, o in list(product(a_list, o_list)):
				a_id, o_id = a_list.index(a), o_list.index(o)
				(i1,i2,p) = updates(i,a,o, para, dim_u)
				key = ((i), (a,o))
				y1[i1] += max(y0[i]*p_dict[key]*outmap[i,a_id]*p, 0)
				y1[i2] += max(y0[i]*p_dict[key]*outmap[i,a_id]*(1-p), 0)
		eR_0 = gen_eR(y0,p_dict,dim_u,para)
		eR_1 = gen_eR(y1/np.sum(y1),p_dict,dim_u,para)
		err = np.abs(eR_1-eR_0)
		y0 = y1/np.sum(y1)
		t += 1
	return np.around(eR_1,6), y0, t

def run_bdpB(para, dim_u=180):
	h, dp, dpm = para
	u_ub = ((1-2*h) + ((1+dpm)/dp*h)**2)**.5 - (1+dpm)/dp*h
	u_list = [-u_ub + 2*u_ub*i/(dim_u-2-1) for i in range(dim_u-2)]
	u0, u1 = u_list[0] - np.diff(u_list).mean(), u_list[-1] + np.diff(u_list).mean()
	u_list = [u0] + u_list + [u1]
	p_dict = gen_p_dict(u_list, para, dim_u)

	# build markov transition matrix
	T = np.zeros([dim_o,dim_u, dim_o,dim_u])
	for i,u in enumerate(u_list):
		a = np.sign(u)
		#
		o, oi = -1, 0
		j1,j2,p = updates(i,a,o, para, dim_u)
		T[oi,i, 0,j1] = p * 	p_dict[(j1, (np.sign(u_list[j1]),-1))]
		T[oi,i, 0,j2] = (1-p) * p_dict[(j2, (np.sign(u_list[j2]),-1))]
		T[oi,i, 1,j1] = p * 	p_dict[(j1, (np.sign(u_list[j1]),1))]
		T[oi,i, 1,j2] = (1-p) * p_dict[(j2, (np.sign(u_list[j2]),1))]
		#
		o, oi = 1, 1
		j1,j2,p = updates(i,a,1, para, dim_u)
		T[oi,i,0,j1] = p *	   p_dict[(j1, (np.sign(u_list[j1]),-1))]
		T[oi,i,0,j2] = (1-p) * p_dict[(j2, (np.sign(u_list[j2]),-1))]
		T[oi,i,1,j1] = p * 	   p_dict[(j1, (np.sign(u_list[j1]),1))]
		T[oi,i,1,j2] = (1-p) * p_dict[(j2, (np.sign(u_list[j2]),1))]
	T = T.reshape(dim_o*dim_u, dim_o*dim_u)

	# find top eigen vector
	w, v = LA.eig(T.T)
	id_top = np.argmax(w)
	v_top = np.abs(v[:,id_top].reshape(dim_o, dim_u).sum(0))
	v_top = v_top/v_top.sum()

	# eR
	eR = gen_eR(v_top,p_dict,dim_u,para)

	# plot
	# eR, p_arr, t_iter = run_bdpB(para, dim_u)
	# plt.figure(figsize=(10,4), dpi=300)
	# plt.plot(p_arr, alpha=.5)
	# plt.plot(v_top, alpha=.5)
	return eR, v_top, -1 # t_iter = -1 is just there to be compartible with run_bdp_v1


##%% bdpB v1
def gen_p_dict_v1(u_list, para, dim_u):
	h, dp, dpm = para
	u_ub = 1#((1-2*h) + ((1+dpm)/dp*h)**2)**.5 - (1+dpm)/dp*h
	u_list = [-u_ub + 2*u_ub*i/(dim_u-1) for i in range(dim_u)]
	y = {}
	for i in range(dim_u):
		du = u_list[i]
		for a, o in list(product(a_list, o_list)):
			key = ((i), (a,o))
			p = (1 + o*(a*dp*du + dpm)) / 2
			if (p>1) or (p<0): y[key] = 0
			else: y[key] = p
	return y

def gen_eR_v1(p_array,p_dict,dim_u,para):
	h, dp, dpm = para
	u_ub = 1#((1-2*h) + ((1+dpm)/dp*h)**2)**.5 - (1+dpm)/dp*h
	u_list = [-u_ub + 2*u_ub*i/(dim_u-1) for i in range(dim_u)]
	outmap = np.array([np.array(u_list)<0, np.array(u_list)>0]).T # rewardseek
	p_uo_list = []
	for i in range(dim_u):
		pu, du = p_array[i], u_list[i]
		for a in a_list:
			a_id = a_list.index(a)
			key = ((i), (a,1))
			p_oa = p_dict[key] * outmap[i,a_id]
			p_uo_list.append(pu * p_oa)
	return np.sum(p_uo_list)

def updates_v1(i,a,o, para, dim_u):
	h, dp, dpm = para
	u_ub = 1#((1-2*h) + ((1+dpm)/dp*h)**2)**.5 - (1+dpm)/dp*h
	u_list = [-u_ub + 2*u_ub*i/(dim_u-1) for i in range(dim_u)]
	a_id, o_id = a_list.index(a), o_list.index(o)
	du = u_list[i]
	du1 = (1-2*h) * (a*o*dp + (1+o*dpm)*du) / (a*o*dp*du + 1+o*dpm + 1e-12)
	i1, i2 = np.argsort(np.abs(du1 - np.array(u_list)))[:2]
	p = (u_list[i2] - du1) / (u_list[i2] - u_list[i1])
	return (i1,i2,p)

def ToStationaryDistribution_v1(p_array,outmap,p_dict,para,dim_u,err_thres=err_thres):
	h,dp,dpm = para
	u_ub = 1#((1-2*h) + ((1+dpm)/dp*h)**2)**.5 - (1+dpm)/dp*h
	u_list = [-u_ub + 2*u_ub*i/(dim_u-1) for i in range(dim_u)]
	y0 = p_array.copy()
	t,err,eR = 0,1,1
	while (err>err_thres) & (t<max_iter):
		y1 = np.zeros(dim_u)
		for i in range(dim_u):
			if y0[i] < .01/dim_u: continue
			for a, o in list(product(a_list, o_list)):
				a_id, o_id = a_list.index(a), o_list.index(o)
				(i1,i2,p) = updates_v1(i,a,o, para, dim_u)
				key = ((i), (a,o))
				y1[i1] += max(y0[i]*p_dict[key]*outmap[i,a_id]*p, 0)
				y1[i2] += max(y0[i]*p_dict[key]*outmap[i,a_id]*(1-p), 0)
		eR_0 = gen_eR_v1(y0,p_dict,dim_u,para)
		eR_1 = gen_eR_v1(y1/np.sum(y1),p_dict,dim_u,para)
		err = np.abs(eR_1-eR_0)
		y0 = y1/np.sum(y1)
		t += 1
	return np.around(eR_1,6), y0, t

def run_bdpB_v1(para, dim_u=200):
	h,dp,dpm = para
	u_ub = 1#((1-2*h) + ((1+dpm)/dp*h)**2)**.5 - (1+dpm)/dp*h
	u_list = [-u_ub + 2*u_ub*i/(dim_u-1) for i in range(dim_u)]
	outmap = np.array([np.array(u_list)<0, np.array(u_list)>0]).T # rewardseek
	p0_array = np.ones(dim_u)/dim_u
	p_dict = gen_p_dict_v1(u_list, para, dim_u)
	eR, p_array, t_iter = ToStationaryDistribution_v1(p0_array,outmap,p_dict,para,dim_u)
	# dat = [eR, p_array, t_iter]
	return eR, p_array, t_iter




# ##%% RUN
# # para = pickle.load(open('data/df_param', 'rb')).loc[14, ['h','dp','dpm']].values
# dim_u = 180
# h = .365
# dp, dpm = .3, -.5
# para = h, dp, dpm
# # eR, p_arr, t_iter = run_bdpB_v1(para)
# eR, p_arr, t_iter = run_bdpB(para, dim_u)
#
# u_ub = ((1-2*h) + ((1+dpm)/dp*h)**2)**.5 - (1+dpm)/dp*h
# u_ub = u_ub*1.
# u_list = [-u_ub + 2*u_ub*i/(dim_u-1) for i in range(dim_u)]
#
# plt.title(eR)
# plt.plot(u_list, p_arr)
