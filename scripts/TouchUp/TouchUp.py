import numpy as np
import itertools
import sys
import os
import Code.Utils.Read_MFPs as Read_MFPs
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
Input_folder = params.Input_folder

MFP_Folder = os.path.join(Input_folder,'Final_Results','MF_Solutions')
Initial_mpfs = Read_MFPs.Read_MFPs(MFP_Folder)

Results_Folder =  os.path.join(Input_folder,'TouchUp')

if not os.path.exists(Results_Folder):
    os.makedirs(Results_Folder)
    os.makedirs(os.path.join(Results_Folder,'MF_Solutions'))

########## Code
a = time()
Model = Hamiltonian(params.Model_Params)
Solver = HFA_Solver(Model,method=params.method,beta= params.beta, Itteration_limit=params.Itteration_limit, tol=params.tolerance)

sweeper = Phase_Diagram_Sweeper(Model,Solver,Initial_mpfs,params.U_values,params.J_values,n_threads,verbose=params.verbose)

sweeper.Sweep()
sweeper.save_results(Results_Folder,Include_MFPs=True)

DiagramPlots(Results_Folder,Model.Dict)

print('time to complete (s):',round(time()-a,3),'Converged points:',round(sweeper.Convergence_pc,3),'%' '\n')
