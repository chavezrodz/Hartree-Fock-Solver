import numpy as np
import itertools
import sys
import os
from Code.HFA_Solver import *
from Code.PhaseDiagramSweeper import *
from Code.Interpreter import Interpreter
from Hamiltonians.Hamiltonian_8 import *
from Utils.tuplelist import *
from time import time

# Input trials folder, output final MFP's


Model_Params = dict(
N_Dim = 2,
N_cells = 10,
Filling = 0.25,
mat_dim = 8,

eps = 1,
t_1 = 1,
U = 1,
J = 1)

U_values = np.linspace(0,6,20)
J_values = np.linspace(0,3,20)

deltas = np.linspace(0,10,5)
sfm = np.linspace(0,10,5)
safm = np.linspace(0,10,5)

params_list = tuplelist([deltas,sfm,safm])

MF_params = params_list[6]

Model = Hamiltonian(Model_Params, MF_params)
Solver = HFA_Solver(Model)

Input_Folder = 'Results/Guesses_Results/'

Optimal_guesses, Optimal_Energy = Interpreter(Input_Folder, params_list)

sweeper = Phase_Diagram_Sweeper(Model,Solver,Optimal_guesses,U_values,J_values)

Final_Results_Folder = os.path.join('Results','Final_Results')

sweeper.Sweep(Final_Results_Folder, Final_Run=True)

Final_Energy = sweeper.Es_trial
print(np.sum(Final_Energy))
print(np.sum(Optimal_Energy))
print(np.array_equal(Final_Energy, Optimal_Energy))