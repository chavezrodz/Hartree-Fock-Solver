import numpy as np
import itertools
import sys
import os
from Code.Solver.HFA_Solver import HFA_Solver
from Code.Solver.PhaseDiagramSweeper import Phase_Diagram_Sweeper
from Code.Nickelates.Hamiltonian import Hamiltonian
# from Utils.tuplelist import *
# from Code.Utils.tuplelist import tuplelistfrom Utils.DispersionRelation import *
from time import time
import params
import argparse

######### Command Line Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_threads', type=int, default = 8)
parser.add_argument('--trial_ind',type=int, default = 0)
args = parser.parse_args()
n = args.trial_ind
n_threads = args.n_threads
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

########## Code
a = time()
MF_params = np.array(params_list[n])

Model = Hamiltonian(Model_Params, MF_params)
Solver = HFA_Solver(Model, beta=beta, Itteration_limit=Itteration_limit, tol=tolerance)

sweeper = Phase_Diagram_Sweeper(Model,Solver,MF_params,U_values,J_values,n_threads,verbose=verbose)

if not os.path.exists(os.path.join(params.Results_Folder,'logs')):
	os.makedirs(os.path.join(params.Results_Folder,'logs'))

Guess_Name = 'Guess'+str(MF_params)

outfolder = os.path.join(params.Results_Folder,'Guesses_Results',Guess_Name)
if not os.path.exists(outfolder):
	os.makedirs(outfolder)

sweeper.Sweep()
sweeper.save_results(outfolder)
print('\n Diagram itteration:',n, 'time to complete (s):',round(time()-a,3),'Converged points:',sweeper.Convergence_pc,'%' '\n')