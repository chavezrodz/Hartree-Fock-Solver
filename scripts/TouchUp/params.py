import os
import numpy as np
import Code.Utils.Read_MFPs as Read_MFPs
########### Model Params
Model_Params = dict(
N_Dim = 2,
Nx = 50,
Ny = 50,
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

########### Solver params
alpha = 100
beta = 0.500001 
gamma = 3
Itteration_limit = 50
tolerance = 1e-3
method = 'sigmoid'
########## Sweeper params
verbose = True
 
Input_folder = os.path.join('Results','Run_17_sept_2')
# Input_folder = os.path.join('Results','Run_sep_15','Final_Results')

MFP_Folder = os.path.join(Input_folder,'MF_Solutions')
Initial_mpfs = Read_MFPs.Read_MFPs(MFP_Folder)

Results_Folder = "Results/TestTouchup"
