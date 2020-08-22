import numpy as np
import itertools
import sys
import os
from Code.HFA_Solver import *
from Code.PhaseDiagramSweeper import *
from Hamiltonians.Hamiltonian_Nickelates import *
from Utils.tuplelist import *
from time import time

Model_Params = dict(
N_Dim = 2,
Nx = 25,
Ny = 25,
Filling = 0.25,
mat_dim = 8,

eps = 0,
t_1 = 1,
t_2 = 0.5,
t_4 = 0,
U = 1,
J = 1)

U_values = np.linspace(0,6,10)
J_values = np.linspace(0,3,10)

deltas = np.linspace(0,1,3)
sfm = np.linspace(0,1,3)
safm = np.linspace(0,1,3)
Deltas = np.linspace(0,1,3)
params_list = tuplelist([deltas,sfm,safm,Deltas])

# if  len(sys.argv)!=2:
#     print("Expected input")
#     exit(2)

# n = int(sys.argv[1])

for n in range(51,81):
# n = 0
	n_threads = 8

	a = time()
	MF_params = np.array(params_list[n])

	Model = Hamiltonian(Model_Params, MF_params)
	Solver = HFA_Solver(Model, beta=0.500001, Itteration_limit=100, tol=1e-2)


	sweeper = Phase_Diagram_Sweeper(Model,Solver,MF_params,U_values,J_values,n_threads,verbose=True)

	fname = str(MF_params)+'.csv'

	outfolder = os.path.join('Results','Guesses_Results')

	sweeper.Sweep(outfolder,fname)

	print('\n Diagram itteration:',n, 'time to complete (s):',round(time()-a,3),'Converged points:',sweeper.Convergence_pc,'%' '\n')	