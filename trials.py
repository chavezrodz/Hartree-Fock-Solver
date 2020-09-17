import numpy as np
import itertools
import sys
import os
from Code.Solver.HFA_Solver import HFA_Solver
from Code.Solver.PhaseDiagramSweeper import Phase_Diagram_Sweeper
from Code.Nickelates.Hamiltonian import Hamiltonian
from time import time
import params
import argparse

######### Command Line Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_threads', type=int, default = 8)
parser.add_argument('--trial_ind',type=int, default = 0)
args = parser.parse_args()

############ Guesses Input
params_list = params.params_list
MF_params = np.array(params_list[args.trial_ind])

Guess_Name = 'Guess'+str(MF_params)

outfolder = os.path.join(params.Results_Folder,'Guesses_Results',Guess_Name)
if not os.path.exists(outfolder):
	os.makedirs(outfolder)

if not os.path.exists(os.path.join(params.Results_Folder,'logs')):
	os.makedirs(os.path.join(params.Results_Folder,'logs'))

########## Code
a = time()

Model = Hamiltonian(params.Model_Params, MF_params)
Solver = HFA_Solver(Model, beta=params.beta, Itteration_limit=params.Itteration_limit, tol=params.tolerance)
sweeper = Phase_Diagram_Sweeper(Model,Solver,MF_params,params.U_values,params.J_values,args.n_threads,verbose=params.verbose)

sweeper.Sweep()
sweeper.save_results(outfolder)
print('\n Diagram itteration:',args.trial_ind, 'time to complete (s):',round(time()-a,3),'Converged points:',round(sweeper.Convergence_pc,3),'%' '\n')