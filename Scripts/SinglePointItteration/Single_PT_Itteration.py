from time import time
import numpy as np
from Code.Solver.HFA_Solver import HFA_Solver
import Code.Display.Itteration_sequence as IS
import Code.Display.DispersionRelation as DR
from Code.Nickelates.Hamiltonian import Hamiltonian

Model_Params = dict(
    N_shape=(15, 15, 15),
    b=0
    )

# Code
a = time()

MF_params = np.array([0,  0.004, 0.477,  0,   0])

method = 'momentum'; beta = 0.5
Model = Hamiltonian(Model_Params, MF_params)
Solver = HFA_Solver(Model, method=method, beta=beta,
                    Itteration_limit=50, tol=1e-3, save_seq=True)
Solver.Itterate(verbose=True)

# DR.DispersionRelation(Solver)
# DR.DispersionRelation(Solver)
# IS.Itteration_sequence(Solver)
# DR.fermi_surface(Solver, save=False)
# print(Solver.Conductor)
DR.DOS(Solver, bins=30)


# print(Solver.bandwidth_calculation())

"""
# Represenntative settings to show adaptive time steps work
MF_params = np.array([0.22,  0.5, -0.5,  0.92,   0.046])
method = 'momentum'; beta = 0.500001
Model = Hamiltonian(Model_Params, MF_params)
Solver = HFA_Solver(Model, method=method, beta=beta, Itteration_limit=50, tol=1e-3, save_seq=True)
Solver.Itterate(verbose=True)

# IS.Itteration_sequence(Solver)

Solver_fixed = Solver
method = 'sigmoid'; beta = 3

Model = Hamiltonian(Model_Params, MF_params)
Solver = HFA_Solver(Model, method=method, beta=beta, Itteration_limit=50, tol=1e-3, save_seq=True)
Solver.Itterate(verbose=True)

Solver_scheduled = Solver

IS.Itteration_comparison(Solver_fixed, Solver_scheduled)
"""
