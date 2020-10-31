import numpy as np
import itertools
import sys
import os
from time import time
import argparse
import logging

from Code.Solver.HFA_Solver import HFA_Solver
from Code.Utils.tuplelist import tuplelist as tp
from Code.Solver.PhaseDiagramSweeper import Phase_Diagram_Sweeper
from Code.Solver.E_Tracker import E_Tracker
from Code.Nickelates.Hamiltonian import Hamiltonian
from Code.Solver.Optimizer import Optimizer_exhaustive as Optimizer_exhaustive
import Code.Solver.Optimizer
from Code.Utils.Read_MFPs import Read_MFPs
from Code.Display.ResultsPlots import E_Plots

Model_Params = dict(
N_shape = (5,5),
Filling = 0.25,
stress=-1,
BZ_rot=1,
eps = 0,
t_1 = 1,
t_2 = 0.15,
t_4 = 0.05,
U = 3,
J = 1)

i = 'J'
i_values = np.linspace(0,3,10)

params_list =[
(1,1,0,1,0.15),
(1,0.5,0,1,0.15),
(0,0.2,0.5,0,0),
(0.1,0.5,1,0.5,0.1),
(0.5,0.5,0,0.5,0.1),
(0.5,0.5,0.5,0.5,0.5)
]

method ='sigmoid'
beta = 1.5
Itteration_limit = 50
tolerance = 1e-3
bw_norm = True

verbose = True

######### Command Line Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_threads', type=int, default = 8)
parser.add_argument('--run_ind',type=int, default=5)
args = parser.parse_args()

outfolder = 'Results/E_Tracker'
if not os.path.exists(outfolder): os.mkdir(outfolder)

Model = Hamiltonian(Model_Params)
Solver = HFA_Solver(Model,method=method,beta=beta, Itteration_limit=Itteration_limit, tol=tolerance)
sweeper = E_Tracker(Model,Solver,i,i_values,guesses=params_list, n_threads=8, Bandwidth_Normalization=bw_norm, verbose=verbose)

sweeper.Sweep()
sweeper.save_results(outfolder,Include_MFPs=True)
E_Plots(i,i_values,Model.Dict,params_list,outfolder)
print('done')
