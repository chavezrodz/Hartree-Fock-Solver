from itertools import product
import itertools
import numpy as np
import os
import glob



# Input list of arrays of energy across phase region, return best guess per region

filelist = sorted(glob.glob('Results/Guesses_Results/*csv'))

N_files = len(filelist)

# Stack all energies arrays
Arrays = []
for file in filelist:
	Arrays.append(np.loadtxt(file,delimiter=','))
a = np.stack(Arrays,axis=0)

# Find Indices of lowest energies across stack
ind = np.unravel_index(np.argmin(a, axis=0), a.shape )[2]

# Recover best guess across phase diagram
u = np.arange(5)
j = np.arange(5)

uj_itterator = itertools.product(u,j)


Optimal_Guesses = np.zeros((5,5,3))

Delta_range = np.arange(2,10,1)
SFM_range = np.arange(2,10,1)
SAFM_range = np.arange(2,10,1)

MFP_itterator = product(Delta_range,SFM_range,SAFM_range)

params_list = []
for i in MFP_itterator:
	params_list.append(i)

for i , v in enumerate(uj_itterator):
	print('U value:',v[0],'J value:',v[1],'Best guess:', params_list[ind[v]] )
	Optimal_Guesses[v] = np.array(params_list[ind[v]])



