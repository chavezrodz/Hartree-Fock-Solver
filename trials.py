import numpy as np
import itertools
import sys
import os
from Code.HFA_Solver import *
from Code.PhaseDiagramSweeper import *
from Hamiltonians.Hamiltonian_Nickelates import *
from Utils.tuplelist import *
from Utils.DispersionRelation import *
from time import time
import params


######### Command Line Arguments
# if  len(sys.argv)!=2:
#     print("Expected input")
#     exit(2)

# n = int(sys.argv[1])
n_threads = params.n_threads
########### Model Params
Model_Params = params.Model_Params
########### Diagram Ranges
U_values = params.U_values
J_values = params.J_values
############ Guesses list
params_list = params.params_list

########### Solver params
beta = params.beta 
Itteration_limit = params.Itteration_limit 
tolerance = params.tolerance
########## Sweeper params
verbose = params.verbose


########## Code

for n in range(len(params_list)):
	a = time()
	MF_params = np.array(params_list[n])

	Model = Hamiltonian(Model_Params, MF_params)
	Solver = HFA_Solver(Model, beta=beta, Itteration_limit=Itteration_limit, tol=tolerance)

	sweeper = Phase_Diagram_Sweeper(Model,Solver,MF_params,U_values,J_values,n_threads,verbose=verbose)

	if not os.path.exists(os.path.join(params.Results_Folder,'logs')):
		os.makedirs(os.path.join(params.Results_Folder,'logs'))

	fname = 'Guess'+str(MF_params)

	outfolder = os.path.join(params.Results_Folder,'Guesses_Results',fname)
	if not os.path.exists(outfolder):
		os.makedirs(outfolder)

	sweeper.Sweep(outfolder)

	print('\n Diagram itteration:',n, 'time to complete (s):',round(time()-a,3),'Converged points:',sweeper.Convergence_pc,'%' '\n')