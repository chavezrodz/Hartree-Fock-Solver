import numpy as np
import itertools
from numpy import linalg as LA
from HFA_Solver import *

class Hamiltonian:
	"""
	Contains: Matrix structure, elements, consistency equations, total energy equation
	and both static and dynamic parameters

	Model_Params must be a dictionary and at least contain:
	N_dim
	N_cells
	Filling
	mat_dim 

	MF_Params must be a 1D np.array

	The class must contain the methods:
	update_variables


	All itterations done in HFA solver.
	"""
	def __init__(self, Model_params, MF_params):
		#initiates Model parameters
		for key, value in Model_params.items():
			setattr(self, key, value)

		#initiates Mean field parameters
		self.MF_params = MF_params
		# self.N_cells = int(self.N_cells*self.Filling)

		self.Qx = np.arange(self.N_cells) #Allowed Momentum values for itterator

	# Static variables, these never change, may depend on momentum indices

		self.U_bar = (3*self.U - 5*self.J)/4
		self.U_0 = (self.U +self.J)/2

		self.tzz = np.zeros((self.N_cells,self.N_cells))
		self.tzz_m1 = np.zeros((self.N_cells,self.N_cells))
		self.tzz_c = np.zeros((self.N_cells,self.N_cells))
		self.tzz_m2 = np.zeros((self.N_cells,self.N_cells))

		qm = np.pi/2 
		qc = np.pi 
		Q = itertools.product(self.Qx,repeat=self.N_Dim)
		for q in Q:
			qx = q[0]*np.pi/self.N_cells - np.pi/2
			qy = q[1]*np.pi/self.N_cells - np.pi/2
			self.tzz[q]    = -2/4*self.t_1*( np.cos(qx)       + np.cos(qy)      )
			self.tzz_m1[q] = -2/4*self.t_1*( np.cos(qx + qm)  + np.cos(qy + qm) )  
			self.tzz_c[q]  = -2/4*self.t_1*( np.cos(qx + qc)  + np.cos(qy + qc) )  
			self.tzz_m2[q] = -2/4*self.t_1*( np.cos(qx - qm)  + np.cos(qy - qm) )  


	def update_variables(self):
		"""
		Calculate dynamic variables
		These depend on MFP, not on momentum
		"""

		# Distortion
		alpha = 1
		beta = 27/4*alpha*self.MF_params[0]**2

		if np.abs(self.MF_params[0]) < 1e-12:
			self.u = 0
		else:
			self.u = 3*self.MF_params[0] / (2*np.cbrt(beta)) * ( np.cbrt(1 + np.sqrt(1 + 1/beta)) + np.cbrt(1 - np.sqrt(1 + 1/beta)) )



	def Mat_q_calc(self,q):
		"""
		Declaration of the matrix to diagonalize, momentum dependent
		"""

		
		# Call static matrix elements
		a00 = self.tzz[q] + self.U_bar 
		a01 = self.tzz_m1[q] + self.U_bar
		a02 = self.tzz_c[q] + self.U_bar
		a03 = self.tzz_m2[q] + self.U_bar


		# could write u = self.u to call dynamic matrix elements
		# Declare sub-block
		sigma = 1

		b = self.U_bar*self.MF_params[0] - 2*self.eps*self.u - sigma * self.U_0*self.MF_params[2]


		a0 = a00 -sigma*self.U_0*self.MF_params[1]
		a1 = a01 -sigma*self.U_0*self.MF_params[1]
		a2 = a02 -sigma*self.U_0*self.MF_params[1]
		a3 = a03 -sigma*self.U_0*self.MF_params[1]

		sub_1 = np.array([
			[a0,0,b,0],
			[0,a1,0,b],
			[b,0,a2,0],
			[0,b,0,a3]])

		# Declare sub-block
		sigma = -1

		b = self.U_bar*self.MF_params[0] - 2*self.eps*self.u - sigma * self.U_0*self.MF_params[2]
		a0 = a00 -sigma*self.U_0*self.MF_params[1]
		a1 = a01 -sigma*self.U_0*self.MF_params[1]
		a2 = a02 -sigma*self.U_0*self.MF_params[1]
		a3 = a03 -sigma*self.U_0*self.MF_params[1]


		sub_2 = np.array([
			[a0,0,b,0],
			[0,a1,0,b],
			[b,0,a2,0],
			[0,b,0,a3]])	


		# Declare matrix
		mat = np.block([
			[sub_1, np.zeros((4,4))], 
			[np.zeros((4,4)),sub_2]
			])

		# Diagonalize Matrix
		w, v  = LA.eig(mat)
		return w, v

	def Consistency(self,v):
		# Consistency Equations, keep order of MFP
		a = ( np.conj(v[0])*v[2] + np.conj(v[1])*v[3] + np.conj(v[2])*v[0] + np.conj(v[3])*v[1] + np.conj(v[4])*v[6] + np.conj(v[5])*v[7] + np.conj(v[6])*v[4] + np.conj(v[7])*v[5])/self.N_cells**2
		b = 0.5*(np.abs(v[0])**2 + np.abs(v[1])**2 + np.abs(v[2])**2 + np.abs(v[3])**2 - np.abs(v[4])**2 - np.abs(v[5])**2 - np.abs(v[6])**2 - np.abs(v[7])**2)/self.N_cells**2
		c = 0.5*( np.conj(v[0])*v[2] + np.conj(v[1])*v[3] + np.conj(v[2])*v[0] + np.conj(v[3])*v[1] - np.conj(v[4])*v[6] - np.conj(v[5])*v[7] - np.conj(v[6])*v[4] - np.conj(v[7])*v[5])/self.N_cells**2
		return a, b, c

	def Calculate_Energy(self):
		E = self.E_occ/self.N_cells**2 + 2*self.eps*(self.u**2/2 + self.u**4/4) - (self.U_bar/2*(1+self.MF_params[0]**2) - self.U_0*(self.MF_params[1]**2 + self.MF_params[2]**2) )/self.N_cells**2
		self.Final_Total_Energy = E
