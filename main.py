from Solver import *

"""
	Need to feed only: 
	Model parameters,
	MFP guesses,
	consistency equations, 
	matrix element equations, 
"""

"""
To Try:
	Gradient Descent with regularization
	Matrix class
"""

N_Dimensions = 2
N_cells = 100

# Dictionaries 
# model parameters
Model_params = dict(
eps = 1,
k_spring = 0.3,
t = 1,
Filling = 0.48
)

# MFP guesses
MFP_params = dict(
delta = 10
)


# Functions
# Consistency Equations
@classmethod
def delta_consistency(cls, v):
	return	0.5*( np.abs(v[0])**2 - np.abs(v[1])**2 )
HFA_solver.delta_consistency = delta_consistency

# Matrix elements

@classmethod
def tb(cls, q):	return -2*Model_params['t']*(
					  np.cos(np.pi*2*q[0]/N_cells) 
					+ np.cos(np.pi*2*q[1]/N_cells))	
HFA_solver.tb = tb

# Initiate Solver
HFA = HFA_solver(N_cells,N_Dimensions,Model_params,MFP_params)

# Itterate
Total_Energy = HFA.Itterate()

# E = HFA.Calculate_occupied_Energy()

# HFA.test()
'''	
if N_Dimensions == 2:
if N_Dimensions == 1:
	@classmethod
	def tb(cls, q):	return -2*Model_params['t']*(
							np.cos(np.pi*q[0]/N_cells))
	HFA_solver.tb = tb

if N_Dimensions == 3:
	@classmethod
	def tb(cls, q):	return -2*Model_params['t']*(
						  np.cos(np.pi*2*q[0]/N_cells) 
						+ np.cos(np.pi*2*q[1]/N_cells)
						+ np.cos(np.pi*2*q[2]/N_cells))	
	HFA_solver.tb = tb
					### Itteration ###
# Compute all eigenvalues
Q = itertools.product(Qx,repeat=N_Dimensions)
for q in Q:
	HFA.MFP_n_calc(q)

# Find all 1/4 lowest states
HFA.Find_filling_lowest_energies()

# Compute new Mean field parameters
a, b = HFA.Calculate_new_del()

print('Initial Mean Field parameters:',a)
print('updated Mean Field parameter:', b)


while np.abs(a - b) > 1e-3:
	Q = itertools.product(Qx,repeat=N_Dimensions)
	for q in Q:
		HFA.MFP_n_calc(q)
	HFA.Find_filling_lowest_energies()
	a, b = HFA.Calculate_new_del()
print('updated Mean Field parameter:', b)

				### End of itteration ###
'''

# print(HFA.occupied_energies)
# print(E)
# print(HFA.indices_array.shape)
# print(HFA.occ_states)
# print(HFA.MFP)
'''

def reader(x):
	if x <1:
		x = 0
		Magnetism = 'ferromagnetic'
	elif not x<1:
		x = 1
		Magnetism = 'antiferromagmetic' 
	print(Magnetism)
	return x

print(reader(HFA.MFP))
'''