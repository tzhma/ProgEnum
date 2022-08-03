'''
NOTE:
1) ToStationaryDistribution has been updated, detecting distribution oscillation and mixing p_arr are implemented
2) BDP is equivalent to computing top eigenvector of joint world MDP & policy from small program
'''
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
import time
from itertools import permutations, combinations, product, islice
import pandas as pd

if not os.path.exists('data'): os.makedirs('data')

## bdp parameters
max_iter = 1000
err_thres = 1e-6
dim_a, dim_o = 2, 2
a_list, o_list = [-1,1], [-1,1] # [a-,a+], [o-,o+]

# dim_u = 1000 ###
# u_list = [-1 + 2*i/(dim_u-1) for i in range(dim_u)]

## functions
def gen_p_dict(para, dim_u):
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
            y[key] = p
    return y

def gen_eR(p_arr,program,p_dict,para,dim_u):
    h,dp,dpm = para
    u_ub = ((1-2*h) + ((1+dpm)/dp*h)**2)**.5 - (1+dpm)/dp*h
    u_list = [-u_ub + 2*u_ub*i/(dim_u-2-1) for i in range(dim_u-2)]
    u0, u1 = u_list[0] - np.diff(u_list).mean(), u_list[-1] + np.diff(u_list).mean()
    u_list = [u0] + u_list + [u1]
    out_map, in_map = program
    dim_m = len(program[0])
    p_uvo_list = []
    for i, k in list(product(range(dim_u),range(dim_m))):
        puv, du, a = p_arr[i,k], u_list[i], out_map[k]
        key = ((i), (a,1))
        p_o = p_dict[key]
        p_uvo_list.append(puv * p_o)
    return np.sum(p_uvo_list)

def updates(i,k,a,o,program,para,dim_u):
    h,dp,dpm = para
    u_ub = ((1-2*h) + ((1+dpm)/dp*h)**2)**.5 - (1+dpm)/dp*h
    u_list = [-u_ub + 2*u_ub*i/(dim_u-2-1) for i in range(dim_u-2)]
    u0, u1 = u_list[0] - np.diff(u_list).mean(), u_list[-1] + np.diff(u_list).mean()
    u_list = [u0] + u_list + [u1]
    out_map, in_map = program
    o_id = o_list.index(o)
    du = u_list[i]
    du1 = (1-2*h) * (a*o*dp + (1+o*dpm)*du) / (a*o*dp*du + 1+o*dpm + 1e-12)
    i1, i2 = np.argsort(np.abs(du1 - np.array(u_list)))[:2]
    k1 = in_map[k][o_id]
    p = (u_list[i2] - du1) / (u_list[i2] - u_list[i1])
    return (i1,i2,p), k1

def ToStationaryDistribution(p_arr,program,p_dict,para,dim_u,num_steps=max_iter):
    h,dp,dpm = para
    u_ub = ((1-2*h) + ((1+dpm)/dp*h)**2)**.5 - (1+dpm)/dp*h
    u_list = [-u_ub + 2*u_ub*i/(dim_u-2-1) for i in range(dim_u-2)]
    u0, u1 = u_list[0] - np.diff(u_list).mean(), u_list[-1] + np.diff(u_list).mean()
    u_list = [u0] + u_list + [u1]
    out_map, in_map = program
    dim_m = len(out_map)
    y0 = p_arr.copy()
    eR_0 = gen_eR(y0,program,p_dict,para,dim_u)
    eR_lis, y_list = [eR_0], [y0]
    for t in range(num_steps):
        y0 = y_list[-1]
        y1 = np.zeros([dim_u, dim_m])
        for (i,k) in list(product(range(dim_u),range(dim_m))):
            if y0[i,k] < .1/(dim_u*dim_m): continue
            du, a = u_list[i], out_map[k]
            for o in [-1,1]:
                (i1,i2,p), k1 = updates(i,k,a,o,program,para,dim_u)
                key = ((i), (a,o))
                y1[i1,k1] += y0[i,k] * p_dict[key] * p
                y1[i2,k1] += y0[i,k] * p_dict[key] * (1-p)
        y1 = y1/np.sum(y1)
        eR_1 = gen_eR(y1,program,p_dict,para,dim_u)
        eR_lis.append(eR_1)
        y_list.append(y1)

        # check convergence of different periodicity
        if t>dim_m:
            err_list = []
            for T in range(1,dim_m+1):
                if t%T==0:
                    eRs_0 = np.array(eR_lis[-2*T:-T])
                    eRs_1 = np.array(eR_lis[-T:])
                    err = np.abs(eRs_1-eRs_0).mean()
                    err_list.append(err)
                    if err<err_thres:
                        eR_1 = np.mean(eRs_1)
                        return np.around(eR_1,6), y1, t
            T_min = np.argmin(err_list) + 1
            y1 = np.stack(y_list[-T_min:]).mean(0)
            y1 = y1/np.sum(y1)
            y_list.append(y1) # override the last
    return np.around(eR_1,6), y1, t

def run_bdp(program, para, dim_u=180):
    h,dp,dpm = para
    u_ub = ((1-2*h) + ((1+dpm)/dp*h)**2)**.5 - (1+dpm)/dp*h
    u_list = [-u_ub + 2*u_ub*i/(dim_u-2-1) for i in range(dim_u-2)]
    u0, u1 = u_list[0] - np.diff(u_list).mean(), u_list[-1] + np.diff(u_list).mean()
    u_list = [u0] + u_list + [u1]
    dim_m = len(program[0])
    # p0_array = (1 + .01*np.random.randn(dim_u,dim_m))/(dim_u*dim_m)
    p0_array = np.ones([dim_u,dim_m])/(dim_u*dim_m)
    p_dict = gen_p_dict(para, dim_u)
    eR, p_arr, t_iter = ToStationaryDistribution(p0_array,program,p_dict,para,dim_u)
    dat = [eR, p_arr, t_iter]
    return dat


# # ##%% RUN
# dim_u = 180
# para = .35, .3, -.5
# df_enumP = pd.read_pickle('data/df_enumP')
# prog = df_enumP.loc[4, 'program'] #WSLG
# eR, p_arr, t_iter = run_bdp(prog, para, dim_u)
# eR
