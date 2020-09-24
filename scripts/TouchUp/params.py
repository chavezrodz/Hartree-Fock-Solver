import os
import numpy as np
import Code.Utils.Read_MFPs as Read_MFPs
########### Model Params
Model_Params = dict(
N_Dim = 2,
Nx = 2,
Ny = 2,
Filling = 0.25,
mat_dim = 8,

eps = 0,
t_1 = 1,
t_2 = 0.15,
t_4 = 0,
U = 1,
J = 1)

############ Diagram Ranges
i = 'U'
j = 'eps'
i_values = np.linspace(0,6,30)
j_values = np.linspace(0,3,30)

########### Solver params
beta = 1
Itteration_limit = 50
tolerance = 1e-3
method = 'sigmoid'
########## Sweeper params
verbose = True
 
Input_folder = os.path.join('Results','Run_sep_20')
# Input_folder = os.path.join('Results','Run_sep_15','Final_Results')
