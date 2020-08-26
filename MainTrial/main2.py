import numpy as np
import itertools
import sys
import os

from Code.HFA_Solver import *
from Code.PhaseDiagramSweeper import *
from Code.Optimizer import Optimizer
from Hamiltonians.Hamiltonian_8 import *
from Utils.tuplelist import *
from time import time

# Input trials folder, output final MFP's

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

Model = Hamiltonian(Model_Params, params_list[1])
Solver = HFA_Solver(Model)

Input_Folder = 'Results/Guesses_Results/'

Optimal_guesses, Optimal_Energy = Optimizer(Input_Folder, params_list)

sweeper = Phase_Diagram_Sweeper(Model,Solver,Optimal_guesses,U_values,J_values)

Final_Results_Folder = os.path.join('Results','Final_Results')

sweeper.Sweep(Final_Results_Folder, Final_Run=True)

Final_Energy = sweeper.Es_trial

print("Initial guess sweep and final calculations are consistent:".np.array_equal(Final_Energy, Optimal_Energy)) 