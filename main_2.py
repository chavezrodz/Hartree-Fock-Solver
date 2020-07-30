import numpy as np
import itertools
from numpy import linalg as LA
from HFA_Solver import *
# model parameters
Model_params = dict(
N_Dim = 2,
N_cells = 5,
Filling = 0.5,
mat_dim = 2,
eps = 1,
t = 1,
k_spring = 1,
)

# MFP guesses
MF_params = [1,1]


class Hamiltonian:
	"""
	Contains: Matrix structure, elements, consistency equations
	and both static and dynamic parameters

	All itterations done in HFA solver.
	"""
	def __init__(self, Model_params, MF_params):
		#initiates Model parameters
		for key, value in Model_params.items():
			setattr(self, key, value)

		#initiates Mean field parameters
		# self.delta = MFP_params['delta']
		self.MF_params = MF_params

		self.Qx = np.arange(self.N_cells) #Allowed Momentum values for itterator

	# Static Matrix Elements

		self.B = np.zeros((self.N_cells,self.N_cells))
		Q = itertools.product(self.Qx,repeat=self.N_Dim)
		for q in Q:
			self.B[q] = -2*self.t*(
						  np.cos(np.pi*2*q[0]/self.N_cells) 
						+ np.cos(np.pi*2*q[1]/self.N_cells))

	def Mat_q_calc(self,q):

		# Call static matrix elements
		b = self.B[q]

		# Calculate dynamic matrix elements
		a = self.MF_params[0] * self.eps**2 / self.k_spring

		# Diagonalize matrix
		mat = np.array([[-a,b],[b,a]])
		w, v  = LA.eig(mat)

		return w, v

	def Consistency(self,v):
		# Consistency Equations, keep order of MFP
		a = 0.5*( np.abs(v[0])**2 - np.abs(v[1])**2 )
		b = 1
		return a, b

	# def Total_Energy(self,indices):
		# E = E_o + 2*


Model = Hamiltonian(Model_params,MF_params)

Sol = HFA_Solver(Model)

Sol.Itterate()
