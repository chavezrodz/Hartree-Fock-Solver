from time import time
import numpy as np
from Code.Solver.HFA_Solver import HFA_Solver
import Code.Display.Itteration_sequence as IS
import Code.Display.DispersionRelation as DR
from Code.Nickelates.Hamiltonian import Hamiltonian

Model_Params = dict(
    N_shape=(25, 25),
    )

# Code
a = time()

# MF_params = np.zeros(5)
MF_params = np.array([0,  0., -0.443,  0,   0])
method = 'sigmoid'; beta = 3
# method = 'momentum'; beta = 0.5
Model = Hamiltonian(Model_Params, MF_params)
Solver = HFA_Solver(Model, method=method, beta=beta,
                    Itteration_limit=250, tol=1e-3, save_seq=True)
Solver.Itterate(verbose=True)

print(time() - a)
# DR.DOS(Solver)
# print(Solver.bandwidth_calculation())

# IS.Itteration_sequence(Solver)
# print(Solver.Conductor)
# DR.DispersionRelation_Zcut(Solver)
DR.fermi_surface_Zcut(Solver, save=False)


"""
# Represenntative settings to show adaptive time steps work
MF_params = np.array([0,  0.004, 0.477,  0,   0])
method = 'momentum'; beta = 0.500001
Model = Hamiltonian(Model_Params, MF_params)
Solver = HFA_Solver(Model, method=method, beta=beta, Itteration_limit=50, tol=1e-3, save_seq=True)
Solver.Itterate(verbose=True)

# IS.Itteration_sequence(Solver)

Solver_fixed = Solver
method = 'sigmoid'; beta = 1.5

Model = Hamiltonian(Model_Params, MF_params)
Solver = HFA_Solver(Model, method=method, beta=beta, Itteration_limit=50, tol=1e-3, save_seq=True)
Solver.Itterate(verbose=True)

Solver_scheduled = Solver

IS.Itteration_comparison(Solver_fixed, Solver_scheduled)
"""
