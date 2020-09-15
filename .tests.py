from time import time
import numpy as np
import itertools
import sys
import os
from Code.HFA_Solver import *
from Code.PhaseDiagramSweeper import *
from Code.Optimizer_touchup import *
from Nickelates.Hamiltonian_Nickelates import *
from Utils.tuplelist import *
from Utils.DiagramPlots import *
from Utils.Read_MFPs import *
import params

########## Command Line Arguments
n_threads = params.n_threads
########### Model Params
Model_Params = params.Model_Params
Dict = params.Dict
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
########## Optimizer params
def Read_MFPs(folder):
	N = len(os.listdir(folder))
	for i in range(N):
		file = os.path.join(folder,'MF'+str(i)+'.csv')
		if i ==0:
			MF = np.loadtxt(file,delimiter=',')
		else:
			MF = np.dstack((MF,np.loadtxt(file,delimiter=',')))
	return MF

MFP_Folder = os.path.join('Results','Results_5mfp','Final_Results','MF_Solutions')
MFPS = Read_MFPs(MFP_Folder)
C_file = os.path.join('Results','Results_5mfp','Final_Results','Convergence_Grid.csv')
Convergence_Grid = np.loadtxt(C_file,delimiter=',')

Final_Results_Folder = 'Results_Test'
print('Initial convergence',np.mean(Convergence_Grid))
optimal_guesses = Optimizer_touchup(MFPS,Convergence_Grid)

Model = Hamiltonian(Model_Params, params_list[1])
Solver = HFA_Solver(Model,beta=beta, Itteration_limit=Itteration_limit, tol=tolerance)

sweeper = Phase_Diagram_Sweeper(Model,Solver,optimal_guesses,U_values,J_values,n_threads,verbose=True)

sweeper.Sweep(Final_Results_Folder, Final_Run=True)

Final_Energies = sweeper.Es_trial

DiagramPlots(Final_Results_Folder,Dict)

# print("Initial guess sweep and final calculations are consistent:",np.array_equal(Final_Energies, Optimal_Energy))
print('time to complete (s):',round(time()-a,3),'Converged points:',sweeper.Convergence_pc,'%' '\n')