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
Nx = 25,
Ny = 25,
Filling = 0.25,
mat_dim = 8,

eps = 0,
t_1 = 1,
t_2 = 0.15,
t_4 = 0,
U = 0.21,
J = 1.55)

########## Code
a = time()
# MF_params = np.zeros(5)
MF_params =np.array([ 0.759 , 0.759 ,-0.082 , 0.682 , 0.089])
# MF_params = np.random.rand(5)*2 -1
# MF_params = np.array([0,-0.18,-0.36,0,0])

Model = Hamiltonian(Model_Params,MF_params)
Solver = HFA_Solver(Model,method='sigmoid',alpha=100, Itteration_limit=100, tol=1e-3)

Solver.Itterate(verbose=True,save_seq=True)

Display_sequence(Solver)
