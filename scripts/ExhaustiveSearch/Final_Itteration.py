import numpy as np
import itertools
import sys
import os
from Code.Solver.HFA_Solver import HFA_Solver
from Code.Solver.PhaseDiagramSweeper import Phase_Diagram_Sweeper
from Code.Nickelates.Hamiltonian import Hamiltonian
from Code.Solver.Optimizer_exhaustive import Optimizer_exhaustive
from Code.Utils.tuplelist import tuplelist
from Code.Display.DiagramPlots import DiagramPlots
from time import time
import argparse
import params

########## Command Line Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_threads', type=int, default = 8)
args = parser.parse_args()
n_threads = args.n_threads

########## Optimizer params
Input_Folder = os.path.join(params.Results_Folder,'Guesses_Results')
Final_Results_Folder = os.path.join(params.Results_Folder,'Final_Results')

if not os.path.exists(Final_Results_Folder):
    os.makedirs(Final_Results_Folder)
    os.makedirs(os.path.join(Final_Results_Folder,'MF_Solutions'))
 
########## Code
a = time()
Model = Hamiltonian(params.Model_Params)
Solver = HFA_Solver(Model,method=params.method, beta= params.beta, Itteration_limit=params.Itteration_limit, tol=params.tolerance)

Optimal_guesses, Optimal_Energy = Optimizer_exhaustive(Input_Folder, params.params_list)

sweeper = Phase_Diagram_Sweeper(Model,Solver,Optimal_guesses,params.U_values,params.J_values,n_threads,verbose=params.verbose)

sweeper.Sweep()
sweeper.save_results(Final_Results_Folder,Include_MFPs=True)

Final_Energies = sweeper.Es_trial

DiagramPlots(Final_Results_Folder,Model.Dict)

print("Initial guess sweep and final calculations are consistent:",np.array_equal(Final_Energies, Optimal_Energy)) 

print('time to complete (s):',round(time()-a,3),'Converged points:',round(sweeper.Convergence_pc,3),'%' '\n')