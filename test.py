from time import time
import numpy as np
import itertools
import sys
import os
import params
import Code.Nickelates.Hamiltonian as Ni
import Code.Solver.HFA_Solver as HFA
import Code.Solver.Optimizer_touchup as ot
import Code.Solver.PhaseDiagramSweeper as Sweeper
import Code.Display.DiagramPlots as Dp
import argparse
import os
import numpy as np
import Code.Utils.Read_MFPs as Read_MFPs

########### Model Params
Model_Params = dict(
N_Dim = 2,
Nx = 25,
Ny = 25,
Filling = 0.25,
mat_dim = 8,

eps = 0,
t_1 = 1,
t_2 = 0.15,
t_4 = 0,
U = 1,
J = 1)

U_values = np.linspace(0,6,30)
J_values = np.linspace(0,3,30)

Dict ={ 0:'Charge Modulation',1:'Ferromagnetism', 2:'Orbital Disproportionation',3:'Anti Ferromagnetism',4:'Antiferroorbital'}

########### Solver params
beta=0.500001
Itteration_limit=1
tol=1e-3

########## Sweeper params
verbose = True
"""
Feed incomplete final results, itterates with nearest neighbours to try and fill the gaps
"""

########## Command Line Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_threads', type=int, default = 8)
args = parser.parse_args()
n_threads = args.n_threads

######### Params file arguments
Model=Ni.Hamiltonian(params.Model_Params,np.array([,,,,]))
Solver = HFA.HFA_Solver(Model,beta=params.beta, Itteration_limit=params.Itteration_limit, tol=params.tol)