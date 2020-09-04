from itertools import product
import itertools
import numpy as np
import os
import glob

def Optimizer(Input_Folder, params_list, verbose=False):
	"""
	Input list of arrays of energy across phase region,
	return best guess per region
	"""

	folderlist = []
	for i in range(len(params_list)):
		folderlist.append(os.path.join(Input_Folder,'Guess'+str(np.array(params_list[i]))))

	# Stack all energies,convergence arrays
	for i, folder in enumerate(folderlist):
		E_file = os.path.join(folder,'Energies.csv')
		C_file = os.path.join(folder,'Convergence_Grid.csv')
		if i ==0:
			E_Tower = np.loadtxt(E_file,delimiter=',')
			C_Tower = np.loadtxt(C_file,delimiter=',')
			Initial_Shape = E_Tower.shape
		else:
			E_Tower = np.dstack((E_Tower,np.loadtxt(E_file,delimiter=',')))
			C_Tower = np.dstack((C_Tower,np.loadtxt(C_file,delimiter=',')))

	# Find Indices of lowest energies across stack
	ind = np.argmin(E_Tower,axis=2)

	# Lowest achievable energy
	Optimal_Energy = np.min(E_Tower,axis=2)
	Optimal_Convergence = np.min(C_Tower,axis=2)

	# Recover best guess across phase diagram
	Optimal_Guesses = np.zeros((*Initial_Shape,len(params_list[0])))

	# Convergence Grid
	u = np.arange(Initial_Shape[0])
	j = np.arange(Initial_Shape[1])
	uj_itterator = itertools.product(u,j)
	for v in uj_itterator:
		if verbose:
			print('U value:',v[0],'J value:',v[1],'Best guess:', params_list[ind[v]] )
		Optimal_Guesses[v] = np.array(params_list[ind[v]])

	return Optimal_Guesses, Optimal_Energy
