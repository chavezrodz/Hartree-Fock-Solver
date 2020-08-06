import numpy as np
import itertools
from numpy import linalg as LA
from multiprocessing import Process
from itertools import product
from multiprocessing import Pool
from time import time


"""
Need to Check:
Parralelize on slurm
print out results in csv
"""

from Hamiltonian_8 import *
# model parameters
Model_Params = dict(
N_Dim = 2,
N_cells = 15,
Filling = 0.25,
mat_dim = 8,

eps = 0,
t_1 = 1,
U = 1,
J = 1)

MF_params = np.array([1,1,1])

################## Phase Diagram ##################

name = str(MF_params)
U_range = 10
J_range = 10


Es_trial = np.zeros((U_range,J_range))
U_range = np.arange(U_range)
J_range = np.arange(J_range)


def Phase_Diagram_point(v):
	Model_Params['U'],Model_Params['J'] = v[0]/10,v[1]/10

	Model = Hamiltonian(Model_Params, MF_params)
	
	Sol = HFA_Solver(Model)
	
	Sol.Itterate(tol=1e-3,verbose=False)
	
	Model.E_occ = Sol.total_occupied_energy

	Model.Calculate_Energy()

	return Model.Final_Total_Energy


Iterator = product(U_range,J_range)

# Energies results to list
a = time()

with Pool(8) as p:
	results = p.map(Phase_Diagram_point, Iterator)
print('Time to complete model parameters sweep:',round(time() - a,ndigits=3), 'seconds')

# Energies results list to array to csv
Iterator = product(U_range,J_range)
for i,v in enumerate(Iterator):
	Es_trial[v] = results[i]
np.savetxt('Guesses/Guess'+name+'.csv',Es_trial,delimiter=',')







# print(E)

# Energies = Sol.Energies
# print(Energies.shape)
# Sol.save_results('8_hamiltonian_Results')


'''
# Fermi Surface
x = Sol.indices_array[0]
y = Sol.indices_array[1]
z = Sol.occupied_energies

plt.scatter(x, y, c=z
	, marker="s"
	)
# plt.colorbar()
plt.show()
'''

'''
from Hamiltonian_2 import *
from HFA_Solver import *
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