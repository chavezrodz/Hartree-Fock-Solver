import numpy as np
import itertools
import sys
import os
from Code.HFA_Solver import *
from Code.PhaseDiagramSweeper import *
from Hamiltonians.Hamiltonian_8 import *
from Utils.tuplelist import *
from time import time

Model_Params = dict(
N_Dim = 2,
Nx = 10,
Ny = 10,
Filling = 0.25,
mat_dim = 8,

eps = 1,
t_1 = 1,
U = 1,
J = 1)

U_values = np.linspace(0,10,20)
J_values = np.linspace(0,10,20)

deltas = np.linspace(0,10,2)
sfm = np.linspace(0,10,2)
safm = np.linspace(0,10,2)

params_list = tuplelist([deltas,sfm,safm])

# if  len(sys.argv)!=2:
    # print("Expected input")
    # exit(2)

n = 7#int(sys.argv[1])
n_threads = 8

for n in range(8):
	a = time()

	MF_params = np.array(params_list[n])

	Model = Hamiltonian(Model_Params, MF_params)
	Solver = HFA_Solver(Model)


	sweeper = Phase_Diagram_Sweeper(Model,Solver,MF_params,U_values,J_values,n_threads)

	fname = str(MF_params)+'.csv'

	outfolder = os.path.join('Results','Guesses_Results')

	sweeper.Sweep(outfolder,fname)
	print('itteration:',n, 'time to complete:',round(time()-a,3))