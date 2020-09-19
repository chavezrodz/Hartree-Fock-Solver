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
Final_Results_Folder = os.path.join(params.Results_Folder,'Final_Results')

if not os.path.exists(Final_Results_Folder):
    os.makedirs(Final_Results_Folder)
    os.makedirs(os.path.join(Final_Results_Folder,'MF_Solutions'))

########## Code
a = time()
Model = Hamiltonian(params.Model_Params)
Solver = HFA_Solver(Model,method=params.method,alpha = params.alpha, beta= params.beta,gamma=params.gamma, Itteration_limit=params.Itteration_limit, tol=params.tolerance)

Optimal_guesses = params.Initial_mpfs

sweeper = Phase_Diagram_Sweeper(Model,Solver,Optimal_guesses,params.U_values,params.J_values,n_threads,verbose=params.verbose)

sweeper.Sweep()
sweeper.save_results(Final_Results_Folder,Include_MFPs=True)

DiagramPlots(Final_Results_Folder,Model.Dict)

print('time to complete (s):',round(time()-a,3),'Converged points:',round(sweeper.Convergence_pc,3),'%' '\n')
