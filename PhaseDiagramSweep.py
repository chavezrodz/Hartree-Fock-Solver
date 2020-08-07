import numpy as np
import itertools
from numpy import linalg as LA
from multiprocessing import Process
from itertools import product
from multiprocessing import Pool
from time import time
from Hamiltonian_8 import *

"""
Need to Check:
print out results in csv ?
"""

# model parameters
Model_Params = dict(
N_Dim = 2,
N_cells = 10,
Filling = 0.25,
mat_dim = 8,

eps = 0,
t_1 = 1,
U = 1,
J = 1)

##################### Creating mfp indice###########
Delta_range = np.arange(2,10,1)
SFM_range = np.arange(2,10,1)
SAFM_range = np.arange(2,10,1)


MFP_itterator = product(Delta_range,SFM_range,SAFM_range)

params_list = []
for i in MFP_itterator:
	params_list.append(i)
####################### Calling MF_params 

import sys
if  len(sys.argv)!=2:
    print("Expected input")
    exit(2)
n = int(sys.argv[1])

MF_params = np.array(params_list[n])


################## Phase Diagram Sweep ##################

name = str(MF_params)

U_range = 5
J_range = 5


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
	print('Itteration Done')
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

##################### End of Phase Diagram Sweep ######################

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