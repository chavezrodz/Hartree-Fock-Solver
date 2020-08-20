import numpy as np
import itertools
import sys
import os

from Code.HFA_Solver import *
from Code.PhaseDiagramSweeper import *
from Code.Interpreter import Interpreter
from Hamiltonians.Hamiltonian_Nickelates import *
from Utils.tuplelist import *
from time import time

# Input trials folder, output final MFP's

Model_Params = dict(
N_Dim = 2,
Nx = 100,
Ny = 100,
Filling = 0.25,
mat_dim = 8,

eps = 1,
t_1 = 1,
t_2 = 1,
t_4 = 1,
U = 1,
J = 1)

U_values = np.linspace(0,10,20)
J_values = np.linspace(0,10,20)

deltas = np.linspace(0,10,10)
sfm = np.linspace(0,10,10)
safm = np.linspace(0,10,10)
Deltas = np.linspace(1,10,10)
params_list = tuplelist([deltas,sfm,safm,Deltas])


n_threads = 28

Model = Hamiltonian(Model_Params, params_list[1])
Solver = HFA_Solver(Model)

Input_Folder = 'Results/Guesses_Results/'

Optimal_guesses, Optimal_Energy = Interpreter(Input_Folder, params_list)

sweeper = Phase_Diagram_Sweeper(Model,Solver,Optimal_guesses,U_values,J_values,n_threads)

Final_Results_Folder = os.path.join('Results','Final_Results')

sweeper.Sweep(Final_Results_Folder, Final_Run=True)

Final_Energy = sweeper.Es_trial

print("Initial guess sweep and final calculations are consistent:".np.array_equal(Final_Energy, Optimal_Energy)) 