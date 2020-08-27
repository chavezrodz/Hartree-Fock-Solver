import os
import numpy as np

########## Command Line Arguments
n_threads = 16
########### Model Params
Model_Params = dict(
N_Dim = 2,
Nx = 35,
Ny = 35,
Filling = 0.25,
mat_dim = 8,

eps = 0,
t_1 = 1,
t_2 = 0.15,
t_4 = 0,
U = 1,
J = 1)
############ Diagram Ranges
U_values = np.linspace(0,6,30)
J_values = np.linspace(0,3,30)

############ Guess ranges

# deltas = np.linspace(0,0.5,1)
# sfm    = np.linspace(0,0.5,1)
# Deltas = np.linspace(0.5,1,1)
# safm   = np.linspace(0,0.5,1)
# params_list = tuplelist([deltas,sfm,safm,Deltas])

params_list = [
(0,0,0.4,0),
(0.7,0.7,0,0.7),
(0,0.2,0.8,0.6),
(1,1,0,1),
(1,1,0.2,0.6),
(0.4,0.4,0.4,0.4),
(0.5,0.1,0.25,0)]

########### Solver params
beta = 0.500001 
Itteration_limit = 500 
tolerance = 1e-3
########## Sweeper params
verbose = True

Results_Folder = "Results"