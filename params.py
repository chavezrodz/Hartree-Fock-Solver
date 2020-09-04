import os
import numpy as np
from Utils.tuplelist import *

########## Command Line Arguments
n_threads = 8
########### Model Params
Model_Params = dict(
N_Dim = 2,
Nx = 10,
Ny = 10,
Filling = 0.25,
mat_dim = 8,

eps = 0,
t_1 = 1,
t_2 = 0.15,
t_4 = 0,
U = 1,
J = 1)
############ Diagram Ranges
U_values = np.linspace(0,6,3)
J_values = np.linspace(0,3,3)

############ Guess ranges
Dict ={ 0:'Charge Modulation',1:'Ferromagnetism', 2:'Orbital Disproportionation',3:'Anti Ferromagnetism'}

deltas = np.linspace(0,1,1)
sfm    = np.linspace(0,1,1)
Deltas_FO = np.linspace(0,1,1)
safm   = np.linspace(0,1,1)
Deltas_AFO = np.linspace(0,1,2)
params_list = tuplelist([deltas,sfm,Deltas_FO,safm,Deltas_AFO])

########### Solver params
beta = 0.500001 
Itteration_limit = 100 
tolerance = 1e-3
########## Sweeper params
verbose = True
Results_Folder = "Results/Results_Test"