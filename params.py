import os
import numpy as np
from Code.Utils.tuplelist import tuplelist
########### Model Params
Model_Params = dict(
N_shape = (5,2),
Filling = 0.25,
stress=0,
eps = 0,
t_1 = 1,
t_2 = 0.15,
t_4 = 0,
U = 1,
J = 1)

############ Diagram Ranges
i,j = 'eps','U',
i_values = np.linspace(0,9,3)
j_values = np.linspace(0,18,3)
############ Guess ranges
"""
deltas = np.linspace(0,1,2)
sfm    = np.linspace(0,1,2)
Deltas_FO = np.linspace(0,1,2)
safm   = np.linspace(0,1,2)
Deltas_AFO = np.linspace(0,1,2)
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

########### Solver params
method ='sigmoid'
beta = 3
Itteration_limit = 50
tolerance = 1e-3

########## Sweeper params
verbose = True
save_guess_mfps = False
Run_ID = 'Itterated:'+str(i)+','+str(j)+'-'
Run_ID = Run_ID+'-'.join("{!s}={!r}".format(key,val) for (key,val) in Model_Params.items())

Results_Folder = os.path.join('Results',Run_ID)
