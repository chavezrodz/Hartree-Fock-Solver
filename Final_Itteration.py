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

# Input trials folder, output final MFP's

Model_Params = dict(
N_Dim = 2,
Nx = 25,
Ny = 25,
Filling = 0.25,
mat_dim = 8,

eps = 0,
t_1 = 1,
t_2 = 0.15,
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

a = time()

n_threads = 8

Model = Hamiltonian(Model_Params, params_list[1])
Solver = HFA_Solver(Model,method='momentum',beta=0.500001, Itteration_limit=100, tol=1e-2)

Input_Folder = 'Results/Guesses_Results/'

Optimal_guesses, Optimal_Energy = Optimizer(Input_Folder, params_list)

sweeper = Phase_Diagram_Sweeper(Model,Solver,Optimal_guesses,U_values,J_values,n_threads,verbose=True)

Final_Results_Folder = os.path.join('Results','Final_Results')

sweeper.Sweep(Final_Results_Folder, Final_Run=True)

Final_Energy = sweeper.Es_trial
print(sweeper.Convergence_Grid)

print("Initial guess sweep and final calculations are consistent:",np.array_equal(Final_Energy, Optimal_Energy)) 
print('time to complete (s):',round(time()-a,3),'Converged points:',sweeper.Convergence_pc,'%' '\n')