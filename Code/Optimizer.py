from itertools import product
import itertools
import numpy as np
import os
import glob
# from Utils.tuplelist import *

A = np.array([
 [[3, 6, 2],
  [9, 2, 6,],
  [8, 8, 7,]],

 [[3, 3, 3],
  [4, 3, 6],
  [2, 8, 8]]])

ind = np.argmin(A,axis=0)

# print(A.shape)
# print(ind)

def Optimize(Input_Folder, params_list):
	"""
	Input list of arrays of energy across phase region,
	return best guess per region
	"""
	# filelist = sorted(glob.glob(Input_Folder+'/*csv'))
	filelist = []
	for i in range(len(params_list)):
		filelist.append(Input_Folder+str(np.array(params_list[i]))+'.csv')
	# Stack all energies arrays
	# print(filelist)
	# print(params_list)

	for i, file in enumerate(filelist):
		if i ==0:
			Arrays = np.loadtxt(file,delimiter=',')
			Initial_Shape = Arrays.shape
		else:
			Arrays = np.dstack((Arrays,np.loadtxt(file,delimiter=',')))

	# Find Indices of lowest energies across stack
	Optimal_Energy = np.min(Arrays,axis=2)

	ind = np.argmin(Arrays,axis=2)
	# Recover best guess across phase diagram
	u = np.arange(Initial_Shape[0])
	j = np.arange(Initial_Shape[1])

	uj_itterator = itertools.product(u,j)

	Optimal_Guesses = np.zeros((*Initial_Shape,len(params_list[0])))

	for i , v in enumerate(uj_itterator):
	# print('U value:',v[0],'J value:',v[1],'Best guess:', params[ind[v]] )
		Optimal_Guesses[v] = np.array(params_list[ind[v]])
	return Optimal_Guesses, Optimal_Energy

