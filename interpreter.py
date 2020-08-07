import itertools
import numpy as np
import os
import glob



# Input list of energies across phase region, return best guess per region



filelist = glob.glob('Guesses/*csv')
N_files = len(filelist)


# Stack all energies arrays
Arrays = []
for file in filelist:
	Arrays.append(np.loadtxt(file,delimiter=','))
a = np.stack(Arrays,axis=0)

# Find Indices of lowest energies across stack
ind = np.unravel_index(np.argmin(a, axis=0), a.shape )[2]

# Recover best guess across phase diagram
u = np.arange(10)
j = np.arange(10)

itterator = itertools.product(u,j)


for i , v in enumerate(itterator):
	print('U value:',v[0],'J value:',v[1],'Best guess:',str(filelist[ind[v]]))



# Missing: Interpret guess file to phase