import numpy as np
import itertools
import sys
import os
from Code.Solver.HFA_Solver import HFA_Solver
from Code.Nickelates.Hamiltonian import Hamiltonian
import matplotlib.pyplot as plt
from time import time
from Code.Display.Display_sequence import Display_sequence

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
U = 0,
J = 2.072)

########## Code
a = time()
# MF_params = np.zeros(5)
MF_params =np.array([ 0.759 , 0.759 ,-0.082 , 0.682 , 0.089])
# MF_params = np.array([ 0.924, -0.934,  0.817, -0.668, -0.02 ])
# MF_params = np.array([ 0.758, -0.712, -0.111, -0.642,  0.056])
# MF_params = np.array([ 0.255, -0.712, -0.384, -0.914,  0.904])
# MF_params = np.array([-0.201,  0.904,  0.169, -0.318, -0.723])
# MF_params = np.array([ 0.708,  0.737, -0.094,  0.647,  0.102])
# MF_params = np.array([ 0.922,  0.995, -0.005,  0.92,   0.046])# + np.random.rand(5) THIS ONE NEVER CONVERGES
# MF_params = np.array([0,-0.18,-0.36,0,0])
# MF_params = np.array([-0.911,  1.,    -0.,    -0.911, -0.055])

# MF_params = np.random.rand(5)*2 -1

Model = Hamiltonian(Model_Params,MF_params)

Solver = HFA_Solver(Model,method='sigmoid', beta=0.5, Itteration_limit=200, tol=1e-3,save_seq=True)

Solver.Itterate(verbose=True)

Display_sequence(Solver)
