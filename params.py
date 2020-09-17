import os
import numpy as np
from Code.Utils.tuplelist import tuplelist

########### Model Params
Model_Params = dict(
N_Dim = 2,
Nx = 75,
Ny = 75,
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
deltas = np.linspace(0,1,3)
sfm    = np.linspace(0,1,3)
Deltas_FO = np.linspace(0,1,3)
safm   = np.linspace(0,1,3)
Deltas_AFO = np.linspace(0,1,3)
params_list = tuplelist([deltas,sfm,Deltas_FO,safm,Deltas_AFO])
"""
params_list =[
(1,1,0,1,0.15),
(1,0.5,0,1,0.15),
(0,0.2,0.5,0,0),
(0.1,0.5,1,0.5,0.1),
(0.5,0.5,0,0.5,0.1),
(0.5,0.5,0.5,0.5,0.5)
]
"""

Dict ={ 0:'Charge Modulation',1:'Ferromagnetism', 2:'Orbital Disproportionation',3:'Anti Ferromagnetism',4:'Antiferroorbital'}
########### Solver params
beta = 0.500001 
Itteration_limit = 150
tolerance = 1e-3
########## Sweeper params
verbose = True
Results_Folder = "Results"
