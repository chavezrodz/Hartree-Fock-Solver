import numpy as np
import itertools
from numpy import linalg as LA
from HFA_Solver import *
from Hamiltonian_2 import *
from Hamiltonian_8 import *
"""
Need to Check:

"""
# model parameters
Model_Params = dict(
N_Dim = 2,
N_cells = 100,
Filling = 0.25,
mat_dim = 8,

eps = 1,
t_1 = 1,
U = 0,
J = 5)

MF_params = np.array([0,0,0])
'''

Model_Params = dict(
N_Dim = 2,
N_cells = 50,
Filling = 0.5,
mat_dim = 2,
eps = 1,
t = 0.1,
k_spring = 1,
)

# MFP guesses
MF_params = np.array([1,1])

'''




Model = Hamiltonian(Model_Params, MF_params)

Sol = HFA_Solver(Model)

Sol.Itterate(tol=1e-3,verbose=True)

Model.E_occ = Sol.total_occupied_energy

# Model.Calculate_Energy()

# print(Model.Final_Total_Energy)

# Energies = Sol.Energies
# print(Energies.shape)
# Sol.save_results('8_hamiltonian_Results')

x = Sol.indices_array[0]
y = Sol.indices_array[1]
z = Sol.occupied_energies

plt.scatter(x, y, c=z
	, marker="s"
	)
# plt.colorbar()
plt.show()
