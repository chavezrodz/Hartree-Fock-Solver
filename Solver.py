import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import itertools

class HFA_solver:
	"""
	Structural Parameters
	N_Cells: Number of unit cells
	N_Dim: Number of Dimensions

	Model_params: Hamiltonian Parameters, must include Filling(float)
	MFP_params: initial guesses for Mean Free Parameters
	"""
	def __init__(self, N_Cells, N_Dim, Model_params,MFP_params):
	# Structural Parameters
		self.N = N_Cells
		self.N_Dim = N_Dim
	
	#initiates Model parameters
		self.Model_params = Model_params
		for key, value in Model_params.items():
			setattr(self, key, value)

	#initiates Mean field parameters
		self.MFP_params = MFP_params	
		for key, value in MFP_params.items():
			setattr(self, key, value)

		# Initiate momentum grid itterator
		N = self.N

	# Create arrays to store MFP values and matrix elements

		mat_dim = 2

		if N_Dim == 1:
			self.Energies = np.zeros((N,mat_dim))
			self.Eigenvectors = np.zeros((N,mat_dim,mat_dim))

			self.B = np.zeros(N)

		elif N_Dim == 2:
			self.Energies = np.zeros((N,N,mat_dim))
			self.Eigenvectors = np.zeros((N,N,mat_dim,mat_dim))

			self.B = np.zeros((N,N))

		elif N_Dim == 3:
			self.Energies = np.zeros((N,N,N,mat_dim))
			self.Eigenvectors = np.zeros((N,N,N,mat_dim,mat_dim))

			self.B = np.zeros((N,N,N))
	
		self.Qx = np.arange(N) #Allowed Momentum values
		
		Q = itertools.product(self.Qx,repeat=self.N_Dim)
		for q in Q:
			self.B[q] = self.tb(q)

		self.n_states = self.Energies.size #Bands x N
		self.occ_states = int(self.Filling*self.n_states)
		self.occupied_energies = np.zeros(self.occ_states)

		for key in self.MFP_params:
			setattr(self, 'sub_'+key, np.zeros(self.occ_states))



	def MFP_n_calc(self,q):
		"""
		Solve HFA equations from previous mean field parameter, returns subparameter
		,eigenvalues and eigenvectors, fixed momentum values
		Warning: eigenvalues not ordered
		"""

		# Call static matrix elements
		b = self.B[q]

		# Calculate dynamic matrix elements
		a = self.delta * self.eps**2 / self.k_spring

		# Diagonalize matrix
		mat = np.array([[-a,b],[b,a]])
		w, v  = LA.eig(mat)

		self.Energies[q] = w
		self.Eigenvectors[q] = v
		return w, v
		
	def Find_filling_lowest_energies(self):
		Initial_shape = self.Energies.shape
		k = self.occ_states
		Ec = self.Energies.flatten()
		indices = np.argpartition(Ec,k)[:k]
		indices = np.unravel_index(indices,shape = Initial_shape)
		self.indices_array = indices
		indices = np.transpose(np.stack(indices))
		self.indices = list(map(tuple,indices))

	def Calculate_new_del(self):
		for i,ind in enumerate(self.indices):
			v = self.Eigenvectors[ind]

			# Self Consistency Equations
			# for /key in MFP_params:
			# for key in self.MFP_params:
				# setattr(self, 'sub_'+key[i], key+_consistency(v))
			self.sub_delta[i] = self.delta_consistency(v)

		prev_delta = self.delta
		self.delta = np.mean(self.sub_delta)
		return prev_delta,self.delta

	def Itterate(self, tol = 1e-3, verbose = False):
		Q = itertools.product(self.Qx,repeat=self.N_Dim)
		for q in Q:
			self.MFP_n_calc(q)

		self.Find_filling_lowest_energies()
		a, b = self.Calculate_new_del()
		count = 1
		print('Initial Mean Field parameters:',a)
		print('Itteration:  1  Mean Field parameter:', b)
		while np.abs(a - b) > tol:
			count += 1
			Q = itertools.product(self.Qx,repeat=self.N_Dim)
			for q in Q:
				self.MFP_n_calc(q)
			self.Find_filling_lowest_energies()
			a, b = self.Calculate_new_del()
			if verbose ==True:
				print('Itteration: ',count,' Mean Field parameter:', b)
		print('Final Mean Field parameter:', b, '\nNumber of itteration steps:', count)
	# def Calculate_occupied_Energy(self):

		for i,ind in enumerate(self.indices):
			# print(i,ind)
			self.occupied_energies[i] = self.Energies[ind]

		total_occupied_energy_states = np.sum(self.occupied_energies)
		return total_occupied_energy_states


	def Calculate_Total_Energy(self):
		""" Calculate total energy from dispersion relation integral and 
		semiclassical energies
		""" 
		Disp_indp_term = self.N * self.eps**2 / (2*self.k_spring)
		Disp_indp_term *= np.square(self.MFP)
		TE = Disp_indp_term + self.Calculate_occupied_Energy()
		return TE