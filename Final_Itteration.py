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
import Code.Utils.Read_MFPs as rm
from time import time
import argparse
import params
import logging

########## Command Line Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_threads', type=int, default = 8)
args = parser.parse_args()
n_threads = args.n_threads


LOG_FOLDER = os.path.join(params.Results_Folder,'logs')
if not os.path.exists(LOG_FOLDER):
	os.makedirs(LOG_FOLDER)

LOG_FILE_NAME = 'logs'+'_final_.txt'
logging.basicConfig(filename=os.path.join(LOG_FOLDER,LOG_FILE_NAME),
			filemode='a+',
			format='%(message)s',
			datefmt='%H:%M:%S',
			level=logging.INFO)
logger = logging.getLogger()
sys.stdout.write = logger.info


########## Optimizer params
Input_Folder = os.path.join(params.Results_Folder,'Guesses_Results')
Final_Results_Folder = os.path.join(params.Results_Folder,'Final_Results')

if not os.path.exists(Final_Results_Folder):
    os.makedirs(Final_Results_Folder)
 
########## Code
a = time()
Model = Hamiltonian(params.Model_Params)
Solver = HFA_Solver(Model,method=params.method, beta= params.beta, Itteration_limit=params.Itteration_limit, tol=params.tolerance)

Optimal_guesses, Optimal_Energy = Optimizer_exhaustive(Input_Folder, params.params_list,input_MFP=params.save_guess_mfps)

sweeper = Phase_Diagram_Sweeper(Model,Solver,Optimal_guesses,params.i,params.i_values,params.j,params.j_values,n_threads,verbose=params.verbose)

sweeper.Sweep()
sweeper.save_results(Final_Results_Folder,Include_MFPs=True)

Final_Energies = sweeper.Es_trial

DiagramPlots(params.i,params.i_values,params.j,params.j_values,Model.Dict,Final_Results_Folder)

print(f'Initial guess sweep and final calculations are consistent:{np.array_equal(Final_Energies, Optimal_Energy}')

print(f'time to complete (s):{round(time()-a,3)} Converged points:{round(sweeper.Convergence_pc,3)} % \n')
