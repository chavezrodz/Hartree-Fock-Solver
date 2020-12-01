from time import time
import numpy as np
from Code.Solver.HFA_Solver import HFA_Solver
from Code.Nickelates.Hamiltonian import Hamiltonian
import Code.Display.Itteration_sequence as IS
import Code.Display.DispersionRelation as DR

Model_Params = dict(
    N_shape=(25, 25),
    Filling=0.25,
    stress=0,
    BZ_rot=1,
    Delta_CT=0,
    eps=0.5,
    t_1=1,
    t_2=0.15,
    t_4=0.05,
    U=1.5,
    J=1.2)

# Code
a = time()

# Represenntative settings to show adaptive time steps work
MF_params = np.array([0.22,  0.5, -0.5,  0.92,   0.046])

method = 'momentum'; beta = 0.500001
Model = Hamiltonian(Model_Params, MF_params)
Solver = HFA_Solver(Model, method=method, beta=beta, Itteration_limit=50, tol=1e-3, save_seq=True)
Solver.Itterate(verbose=True)

# IS.Itteration_sequence(Solver)

"""
Solver_fixed = Solver
method = 'sigmoid'; beta = 3

Model = Hamiltonian(Model_Params, MF_params)
Solver = HFA_Solver(Model, method=method, beta=beta, Itteration_limit=50, tol=1e-3, save_seq=True)
Solver.Itterate(verbose=True)

Solver_scheduled = Solver

IS.Itteration_comparison(Solver_fixed, Solver_scheduled)
"""

DR.DispersionRelation(Solver)
# DR.fermi_surface(Solver, save=False)
# print(Solver.Conductor)
# DR.DOS(Solver)


# print(Solver.bandwidth_calculation())