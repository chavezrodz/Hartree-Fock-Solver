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

		self.N_states = self.Energies.size #Bands x N
		self.N_occ_states = int(Ham.Filling*self.N_states)
		self.N_params = len(Ham.MF_params)
		self.N_digits = int(np.abs(np.log10(self.tol)))
		self.occupied_energies = np.zeros(self.N_occ_states)
		self.sub_params = np.zeros((self.N_params,self.N_occ_states))

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
		b = self.Hamiltonian.MF_params
		a = np.sum(self.sub_params,axis=1)
		return a, b

	def update_method(self,a,b):
		if self.method == 'momentum':
			return (1 - self.beta)*b + self.beta*a
		else:
			print('Error: Itteration Method not found')

	def Print_step(self,a,method=None):
		if method==None:
			print('Itteration:',self.count,' Mean Field parameters:', a.round(self.N_digits))
			return
		elif method=='Initial':
			print('Initial Mean Field parameters:',a.round(self.N_digits))
			return
		elif method=='Final':
			print('Final Mean Field parameter:', a.round(self.N_digits), 'Number of itteration steps:', self.count)
			return

	def Itteration_Step(self,verbose,save_seq):
		# 	Calculate Dynamic Variables
		self.Hamiltonian.update_variables()
		# Solve Matrix Across all momenta
		Q = itertools.product(self.Hamiltonian.Qx,self.Hamiltonian.Qy)
		for q in Q:
			self.Energies[q],self.Eigenvectors[q] = self.Hamiltonian.Mat_q_calc(q)
		# Find Indices of all required lowest energies
		self.Find_filling_lowest_energies()
		# Calculate Mean Field Parameters with lowest energies
		New_MFP, previous_MFP = self.Calculate_new_del()
		# Update Guess
		New_Guess = self.update_method(New_MFP, previous_MFP)
		self.Hamiltonian.MF_params = New_Guess
		# Logging
		self.count += 1
		if verbose:
			self.Print_step(New_MFP)
		if save_seq:
			self.sol_seq[self.count] = New_MFP
		return New_MFP, New_Guess

	def Itterate(self, verbose=True, save_seq=False):
		self.count = 0
		self.converged = True

		c = self.Hamiltonian.MF_params
		if verbose:
			self.Print_step(c,method='Initial')
		if save_seq:
			self.sol_seq = np.zeros((self.Itteration_limit+1,self.N_params))
			self.sol_seq[0] = c

		a,b = self.Itteration_Step(verbose,save_seq)

		while np.sum(np.abs(a-c)) > self.tol:
			c = b
			a,b = self.Itteration_Step(verbose,save_seq)
			if self.count >= self.Itteration_limit:
				self.converged = False
				break

		if verbose:
			self.Print_step(a,method='Final')

		for i,ind in enumerate(self.indices):
			self.occupied_energies[i] = self.Energies[ind]

		self.total_occupied_energy =  np.sum(self.occupied_energies)
		self.Fermi_Energy = np.max(self.occupied_energies)

	def Metal_or_insulator(self):
		pass