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
import sparse
from core_enum import fl


## sdp parameters
err_thres = 1e-6
dim_a, dim_o = 2, 2
a_list, o_list = [-1,1], [-1,1] # [a-,a+], [o-,o+]


## bdp parameters
max_iter = 1000


## functions: sdp
def gen_A_B(prog, para, dim_u):
    h, dp, dpm = para
    u_ub = ((1-2*h) + ((1+dpm)/dp*h)**2)**.5 - (1+dpm)/dp*h
    u_list = [-u_ub + 2*u_ub*i/(dim_u-2-1) for i in range(dim_u-2)]
    u0, u1 = u_list[0] - np.diff(u_list).mean(), u_list[-1] + np.diff(u_list).mean()
    u_list = [u0] + u_list + [u1]
    outmap, inmap = prog
    dim_m = len(outmap)
    #
    A = np.zeros([dim_u, dim_m, dim_a, dim_o])
    for u,m,a,o in product(range(dim_u), range(dim_m), range(dim_a), range(dim_o)):
        sgn_a, sgn_o = a_list[a], o_list[o]
        pa_m = (sgn_a==outmap[m])*1
        po_ua = (1 + sgn_o*(sgn_a*dp*u_list[u] + dpm)) / 2
        pao_mu = po_ua * pa_m
        A[u,m,a,o] = pao_mu
    #
    B = np.zeros([dim_u, dim_m, dim_u, dim_m, dim_a, dim_o]) # last 4 indices need to align
    for a,o,m,m1 in product(range(dim_a), range(dim_o), range(dim_m), range(dim_m)):
        sgn_a, sgn_o = a_list[a], o_list[o]
        pm1_mo = (m1==inmap[m,o])*1
        if pm1_mo==0: continue
        for u in range(dim_u):
            du1 = (1-2*h) * (sgn_a*sgn_o*dp + (1+sgn_o*dpm)*u_list[u]) / (sgn_a*sgn_o*dp*u_list[u] + 1+sgn_o*dpm + 1e-12)
            u1, u2 = np.argsort(np.abs(du1 - np.array(u_list)))[:2]
            pu1_uao = (u_list[u2] - du1) / (u_list[u2] - u_list[u1])
            pu2_uao = 1 - pu1_uao
            pmu1_muao = pu1_uao * pm1_mo
            pmu2_muao = pu2_uao * pm1_mo
            B[u,m,u1,m1,a,o] = pmu1_muao
            B[u,m,u2,m1,a,o] = pmu2_muao
    return A, B

def update_X_Y(X0, Y0, A, B):
    # X0 = np.ones([dim_u,dim_m,2,2,2,2])
    AX0 = A[:,:,:,:,None] * X0[:,:,None,None] # NEED RENORM
    AX0 = AX0 / (AX0.sum(0).sum(0)[None,None] + 1e-16)
    BAX0 = B[:,:,:,:,:,:,None] * AX0[:,:,None,None]
    X1 = BAX0.sum(0).sum(0)
    # print(X1.shape)
    X1 = X1.reshape( X1.shape[:2] + (np.prod(X1.shape[2:]),) )
    # print(X1.shape)

    # Y0 = np.ones([2,2,2,2,2,2])
    X1Y0 = X1 * Y0[None,None]
    AX1Y0 = A[:,:,:,:,None] * X1Y0[:,:,None,None]
    Y1 = AX1Y0.sum(0).sum(0)
    Y1 = Y1.flatten()
    return X1, Y1

def run_sdp(p_arr, prog, para, dim_u=180, t_iter_max=5):
    #
    A, B = gen_A_B(prog, para, dim_u)
    A = sparse.COO(A)
    B = sparse.COO(B)
    X_init = p_arr[:,:,None]
    Y_init = (A * p_arr[:,:,None,None]).sum(0).sum(0).flatten()
    #
    X, Y = X_init, Y_init
    Y_list = [Y_init]
    for t in range(t_iter_max-1):
        X, Y = update_X_Y(X, Y, A, B)
        Y_list.append(Y)
    return Y_list

def gen_denom_list(Y_lists, numpy_only=False):
    '''
    this step requires densify sparse_arr, which takes time & memory (3m,77GB)
    '''
    try:
        denom_list = pickle.load(open('data/denom_list','rb'))
        return denom_list

    except FileNotFoundError:
        if numpy_only: Y_arr_list = [np.array([x[t] for x in Y_lists]) for t in range(len(Y_lists[0]))]
        else: Y_arr_list = [np.array([x[t].todense() for x in Y_lists]) for t in range(len(Y_lists[0]))]
        denom_list = [x.sum(0)+1e-16 for x in Y_arr_list]
        pickle.dump(denom_list, open('data/denom_list','wb'))
    return denom_list

def gen_pprog_lists(Y_lists, Y0_list, denom_list):
    '''
    single: 1m30s
    '''
    pprog_lists = []
    for Y1_list in Y_lists:
        Y1Y0_list = [(Y1*Y0/denom).sum() for Y0,Y1,denom in zip(Y0_list, Y1_list, denom_list)]
        pprog_lists.append(Y1Y0_list)
    return pprog_lists

def gen_eR_from_Y_list(Y_list):
    '''
    sanity check:
    generate eR from ao_hist
    '''
    seq_depth = int(np.log2(len(Y_list))/2)
    eR_seqs = (np.array(list(product([-1,1], repeat=seq_depth*2)))[:,1::2]>0).sum(1)/seq_depth
    eR = (Y_list * eR_seqs).sum()
    return eR
