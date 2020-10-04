import os
import numpy as np
from Code.Utils.tuplelist import tuplelist
########### Model Params
Model_Params = dict(
N_shape = (50,50),
Filling = 0.25,
stress=0,
eps = 0.8,
t_1 = 1,
t_2 = 0.15,
t_4 = 0,
U = 1,
J = 1)

############ Diagram Ranges
i,j = 'U','J',
i_values = np.linspace(0,6,30)
j_values = np.linspace(0,3,30)
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
beta = 1.5
Itteration_limit = 250
tolerance = 1e-3

########## Sweeper params
verbose = True
save_guess_mfps = False
Run_ID = 'Itterated:'+str(i)+','+str(j)+'-'
Run_ID = Run_ID+'-'.join("{!s}={!r}".format(key,val) for (key,val) in Model_Params.items())

Results_Folder = os.path.join('Results',Run_ID)
