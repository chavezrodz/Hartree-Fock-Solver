import numpy as np
import itertools
import sys
import os
from Code.HFA_Solver import *
from Code.PhaseDiagramSweeper import *
from Code.Optimizer import Optimizer
from Hamiltonians.Hamiltonian_Nickelates import *
from Utils.tuplelist import *
from time import time

########## Command Line Arguments
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
Final_Results_Folder = os.path.join('Results','Final_Results')
########## Optimizer params
Input_Folder = 'Results/Guesses_Results/'

########## Code
params_list = tuplelist([deltas,sfm,safm,Deltas])


a = time()
Model = Hamiltonian(Model_Params, params_list[1])
Solver = HFA_Solver(Model,beta=beta, Itteration_limit=Itteration_limit, tol=tolerance)

Optimal_guesses, Optimal_Energy = Optimizer(Input_Folder, params_list)

sweeper = Phase_Diagram_Sweeper(Model,Solver,Optimal_guesses,U_values,J_values,n_threads,verbose=True)

sweeper.Sweep(Final_Results_Folder, Final_Run=True)
Final_Energy = sweeper.Es_trial
print(sweeper.Convergence_Grid)

print("Initial guess sweep and final calculations are consistent:",np.array_equal(Final_Energy, Optimal_Energy)) 

print('time to complete (s):',round(time()-a,3),'Converged points:',sweeper.Convergence_pc,'%' '\n')