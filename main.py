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


########## Command Line Arguments

if  len(sys.argv)!=2:
    print("Expected input")
    exit(2)

n = int(sys.argv[1])
 
n_threads = 28


########### Model Params
Model_Params = dict(
N_Dim = 2,
Nx = 75,
Ny = 75,
Filling = 0.25,
mat_dim = 8,

eps = 0,
t_1 = 1,
t_2 = 0.15,
t_4 = 0,
U = 1,
J = 1)
############ Diagram Ranges
U_values = np.linspace(0,6,40)
J_values = np.linspace(0,3,40)
############ Guess ranges
deltas = np.linspace(0,1,4)
sfm    = np.linspace(0,1,4)
safm   = np.linspace(0,1,4)
Deltas = np.linspace(0,1,4)
########### Solver params
beta = 0.500001 
Itteration_limit = 500 
tolerance = 1e-3
########## Sweeper params
verbose = True
outfolder = os.path.join('Results','Guesses_Results')

########## Code
params_list = tuplelist([deltas,sfm,safm,Deltas])
a = time()
MF_params = np.array(params_list[n])

Model = Hamiltonian(Model_Params, MF_params)
Solver = HFA_Solver(Model, beta=beta, Itteration_limit=Itteration_limit, tol=tolerance)

sweeper = Phase_Diagram_Sweeper(Model,Solver,MF_params,U_values,J_values,n_threads,verbose=verbose)

fname = str(MF_params)+'.csv'

sweeper.Sweep(outfolder,fname)

print('\n Diagram itteration:',n, 'time to complete (s):',round(time()-a,3),'Converged points:',sweeper.Convergence_pc,'%' '\n')
