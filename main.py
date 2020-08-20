import numpy as np
import itertools
import sys
import os
from Code.HFA_Solver import *
from Code.PhaseDiagramSweeper import *
from Hamiltonians.Hamiltonian_Nickelates import *
from Utils.tuplelist import *
from time import time

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
Deltas = np.linspace(0,10,10)
params_list = tuplelist([deltas,sfm,safm,Deltas])

if  len(sys.argv)!=2:
    print("Expected input")
    exit(2)

n = int(sys.argv[1])

n_threads = 28

a = time()

Model = Hamiltonian(Model_Params, np.array(params_list[n]))
Solver = HFA_Solver(Model)


sweeper = Phase_Diagram_Sweeper(Model,Solver,MF_params,U_values,J_values,n_threads)

fname = str(MF_params)+'.csv'

outfolder = os.path.join('Results','Guesses_Results')

sweeper.Sweep(outfolder,fname)

print('itteration:',n, 'time to complete:',round(time()-a,3))