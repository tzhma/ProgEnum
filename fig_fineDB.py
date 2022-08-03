'''
run with ox (for higher version of networkx to plot self-loop)
df, df_enumP, mergP_dict has to be globally accessible
mp for 5105: 25s
mp for 268533: 50m
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
from core_enum import fl, reshape_prog, gen_stdz_prog
from core_bdpB import run_bdpB_v1


## FUNCTIONS
def gen_DB(dim_u, para, alpha, fullrange):
	'''
	...
	'''
	h, dp, dpm = para
	outmap = (np.arange(dim_u)>=dim_u//2)*2-1 # symmetrical outmap as rewardseek policy

	# knowledge-based discretization with a flexible range for belief value
	inmap_fl_list = []
	inmap_fl_m0_list = []

	if fullrange: u_list = np.around(np.linspace(-1, 1, dim_u), 6)
	else: u_list = np.around(np.linspace((-1+2*h)*alpha, (1-2*h)*alpha, dim_u), 6)

	# inmap_u is a table of transitioned belief values from discretized grid points
	inmap_u = np.zeros([dim_u, 2])
	for i, (u,a) in enumerate(zip(u_list, outmap)):
		for j,o in enumerate([-1,1]):
			u1 = (1-2*h) * (a*o*dp + (1+o*dpm)*u) / (a*o*dp*u + 1+o*dpm + 1e-12)
			inmap_u[i,j] = u1

	# keeping transitioned m0, m1, p0 for later DB enumeration
	inmap_fl_u = inmap_u.flatten() # belief value after transition from grid points
	inmap_fl_m0m1 = [np.argsort(np.abs(x - np.array(u_list)))[:2] for x in inmap_u.flatten()] # two transitioned states closest to the above belief value
	p0_fl = [(u_list[m[1]]-u) / (u_list[m[1]]-u_list[m[0]]) for (m,u) in zip(inmap_fl_m0m1, inmap_fl_u)] # probability of transitioning into closest state
	inmap_fl_m0m1p0 = [(x[0],x[1],y) for x,y in zip(inmap_fl_m0m1, p0_fl)]

	# most likely inmap
	inmap_fl_m0 = tuple([x[y] for x,y in zip(inmap_fl_m0m1, (0,)*(2*dim_u))])
	inmap = np.array(inmap_fl_m0).reshape([len(outmap), 2])

	return outmap, inmap

def plot_shrimp_DB(para, DBsize=30, alpha=1, stdzP='stdz_outmap', fullrange=True):
	'''
	...
	'''
	h, dp, dpm = para
	fineDB = gen_DB(DBsize, para, alpha, fullrange)

	# load
	outmap_tar, inmap_tar = fineDB

	# node, edge lists
	node_list = range(len(outmap_tar))
	node_linewidth_list = [.5 for x in node_list]
	edge_list = fl([[(i,x,y) for x,y in zip(range(len(outmap_tar)), inmap_tar[:,i])] for i in [0,1]])
	edge_lists = [[x for x in edge_list if x[0]==i] for i in [0,1]] # seperating winning/losing edges

	# add nodes
	G = nx.MultiDiGraph()
	G.add_nodes_from([(x, {'label':y, 'color':('#EEEEEE', '#4A4A4A')[int((y+1)/2)], 'linewidth':w}) for x,y,w in zip(node_list, outmap_tar, node_linewidth_list)])

	# add edges
	for o,s,t in edge_list: # outcome, source, target
		if o==0: style='dashed'
		else: style='solid'

		G.add_edge(s, t, weight=1.5, style=style, color='dimgray')
		G[s][t][0]['alpha']=1
		try: G[s][t][1]['alpha']=1
		except KeyError: pass

	# prepare edge attributes
	pos = [[x*.75,0] for x in range(len(outmap_tar))]
	edges_0 = [x for x in list(G.edges) if x[-1]==0]
	edges_1 = [x for x in list(G.edges) if x[-1]==1]
	colors_0 = [nx.get_edge_attributes(G,'color')[y] for y in edges_0]
	colors_1 = [nx.get_edge_attributes(G,'color')[y] for y in edges_1]
	alphas_0 = [nx.get_edge_attributes(G,'alpha')[y] for y in edges_0]
	alphas_1 = [nx.get_edge_attributes(G,'alpha')[y] for y in edges_1]
	weights_0 = [nx.get_edge_attributes(G,'weight')[y] for y in edges_0]
	weights_1 = [nx.get_edge_attributes(G,'weight')[y] for y in edges_1]
	styles_0 = [nx.get_edge_attributes(G,'style')[y] for y in edges_0]
	styles_1 = [nx.get_edge_attributes(G,'style')[y] for y in edges_1]

	# plot
	fig, ax = plt.subplots(dpi=300)
	plt.title('h, dp, dpm = %s, %s, %s'%(h, dp, dpm))
	nx.draw_networkx_nodes(
		G,
		pos,
		node_size=65*DBsize/36,
		node_color=list(nx.get_node_attributes(G,'color').values()),
		vmin=0,
		vmax=1,
		edgecolors='k',
		linewidths=node_linewidth_list,
	)
	nx.draw_networkx_edges(
		G,
		pos,
		edgelist=edges_0,
		node_size=65*DBsize/36,
		arrowstyle='-|>',
		arrowsize=4,
		edge_color='k',
		alpha=alphas_0,
		width=.3,###
		style=styles_0,
		min_source_margin=0,
		min_target_margin=5,
		connectionstyle='arc3,rad=-.7',
	)
	nx.draw_networkx_edges(
		G,
		pos,
		edgelist=edges_1,
		node_size=65*DBsize/36,
		arrowstyle='-|>',
		arrowsize=4,
		edge_color=colors_1,
		alpha=alphas_1,
		width=.3,
		style=styles_1,
		min_source_margin=0,
		min_target_margin=0,
		connectionstyle='arc3,rad=-.7',
	)
	plt.axis('off')
	ax.set_aspect(.45)
	# ax.set_aspect('auto')
	fig.tight_layout()
	if not os.path.exists('fig'): os.makedirs('fig')
	plt.savefig('fig/fineDB.svg', dpi=300)

	return fineDB

## RUN
para = .05, .3, -.5
eR, p_arr, t_iter = run_bdpB_v1(para)



## PLOT: p(u)
plt.figure(figsize=(6,1.5), dpi=300)
plt.plot(np.linspace(-1,1,len(p_arr)), p_arr, color='k', linewidth=1, zorder=1)
plt.fill_between(np.linspace(-1,1,len(p_arr)), p_arr, color='lightgray', alpha=1, zorder=0)
plt.xticks([])
plt.yticks([])
plt.savefig('fig/pu_B_param h/pu_B_fullrange_h=%s.svg'%para[0])


## PLOT: fine DB
plot_shrimp_DB(para, DBsize=30, alpha=1, stdzP='stdz_outmap')









#%%
# df_branch[(df_branch['num_mut']==3) & (df_branch['d2enumDB']==1) & (df_branch['eR']>0.277911) & (df_branch['progsize']==5)][['target','eR','d2enumDB']][:60]
