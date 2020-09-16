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
beta=1
Itteration_limit=5
tol=1e-3

########## Sweeper params
verbose = True
Input_folder = os.path.join('Results','Run_sep_15','Final_Results')

MFP_Folder = os.path.join(Input_folder,'MF_Solutions')
Initial_mpfs = Read_MFPs.Read_MFPs(MFP_Folder)

C_file = os.path.join(Input_folder,'Convergence_Grid.csv')
Initial_Convergence_Grid = np.loadtxt(C_file,delimiter=',')

outfolder = os.path.join('Results','Results_Test_NN')