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
import params

########## Command Line Arguments
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
########## Optimizer params
Input_Folder = params.Final_Input_Folder
Final_Results_Folder = params.Final_Results_Folder


########## Code
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
