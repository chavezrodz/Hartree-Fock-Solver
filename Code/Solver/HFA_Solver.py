import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import itertools

class HFA_Solver:
	"""
	Structural Parameters
	Model_params: Hamiltonian Parameters, must include Filling(float)
	MFP_params: initial guesses for Mean Free Parameters
	"""
	def __init__(self, Ham, method='momentum', beta=0.7, Itteration_limit=50, tol=1e-3):
		self.Hamiltonian = Ham

		self.Energies = np.zeros((*Ham.N_shape,Ham.mat_dim))
		self.Eigenvectors = np.zeros((*Ham.N_shape,Ham.mat_dim,Ham.mat_dim),dtype=complex)
		
		# Itteration Method Params
		self.beta = beta
		self.Itteration_limit = Itteration_limit
		self.tol = tol
		self.method = method

		self.sol_seq = np.zeros(shape=(Itteration_limit+1,len(Ham.MF_params)))

		self.N_states = self.Energies.size #Bands x N
		self.N_occ_states = int(Ham.Filling*self.N_states)
		self.occupied_energies = np.zeros(self.N_occ_states)
		self.sub_params = np.zeros((len(Ham.MF_params),self.N_occ_states))

	def Find_filling_lowest_energies(self):
		Initial_shape = self.Energies.shape
		k = self.N_occ_states
		Ec = self.Energies.flatten()
		indices = np.argpartition(Ec,k)[:k]
		indices = np.unravel_index(indices,shape = Initial_shape)
		self.indices_array = indices
		indices = np.transpose(np.stack(indices))
		self.indices = list(map(tuple,indices))

	def Calculate_new_del(self):
		for i,ind in enumerate(self.indices):
			v = self.Eigenvectors[ind[:-1]][:,ind[-1]]
			self.sub_params[:,i] = np.real(self.Hamiltonian.Consistency(v))
		a = self.Hamiltonian.MF_params
		b = np.sum(self.sub_params,axis=1)
		return a, b

	def momentum_update(self,a,b):
		return (1 - self.beta)*a + self.beta*b

	def update_guess(self,a,b):
		"""
		allows for different possible updating methods
		"""
		if self.method == 'momentum':
			self.Hamiltonian.MF_params = self.momentum_update(a,b)
		else:
			print('Error: Itteration Method not found')

	def Itteration_Step(self):
		# 	Calculate Dynamic Variables
		self.Hamiltonian.update_variables()
		# Solve Matrix Across all momenta
		Q = itertools.product(self.Hamiltonian.Qx,self.Hamiltonian.Qy)
		for q in Q:
			self.Energies[q],self.Eigenvectors[q] = self.Hamiltonian.Mat_q_calc(q)
		# Find Indices of all required lowest energies
		self.Find_filling_lowest_energies()
		# Calculate Mean Field Parameters with lowest energies
		previous_MFP, New_MFP = self.Calculate_new_del()
		self.update_guess(previous_MFP, New_MFP)
		if self.count==0:
			self.sol_seq[self.count] = previous_MFP
		else:
			self.sol_seq[self.count] = New_MFP

		return previous_MFP, New_MFP

	def Metal_or_insulator(self):
		pass


	def Itterate(self, verbose = True):
		self.count = 0
		self.converged = True

		digits = int(np.abs(np.log10(self.tol)))
		a,b = self.Itteration_Step()
		self.sol_seq[1] = self.sol_seq[0]

		if verbose == True:
			print('Initial Mean Field parameters:',a.round(digits))
			print('Itteration: ',self.count,' Mean Field parameters:', b.round(digits))

		self.count += 1

		a,b = self.Itteration_Step()
		if verbose ==True:
			print('Itteration: ',self.count,' Mean Field parameters:', b.round(digits))
		self.count += 1

		while np.sum(np.abs(b- self.momentum_update(self.sol_seq[self.count-1],self.sol_seq[self.count -2]))) > self.tol:
			a,b = self.Itteration_Step()
			if verbose ==True:
				print('Itteration: ',self.count,' Mean Field parameters:', b.round(digits))
			if self.count == self.Itteration_limit:
				self.converged = False
				break
			self.count += 1
				# print("\t \t \t Warning! Did not converge")

		if verbose == True:
			print('Final Mean Field parameter:', b.round(digits), '\nNumber of itteration steps:', self.count)

		for i,ind in enumerate(self.indices):
			self.occupied_energies[i] = self.Energies[ind]

		self.total_occupied_energy =  np.sum(self.occupied_energies)
		self.Fermi_Energy = np.max(self.occupied_energies)