import numpy as np
import itertools
import sys
from numpy import linalg as LA
from multiprocessing import Process
from itertools import product
from multiprocessing import Pool
from time import time
import os

"""
1D sweeper to track state's energies

"""


class E_Tracker:
	"""
	"""

	def __init__(self, Model, Solver, Interpreter, U_values, states, n_threads=8,verbose = False):
		self.Model = Model
		self.Solver = Solver
		
		self.U_values = U_values
		self.U_idx = np.arange(len(U_values))

		self.n_threads = n_threads
		self.verbose = verbose
		
		if Initial_params.ndim == 1:
			self.Initial_params = np.zeros((len(self.U_values),len(self.J_values),len(Model.MF_params)))
			self.Initial_params[:,:,:] = Initial_params
		else:
			self.Initial_params = Initial_params

		self.Es_trial = np.zeros((len(self.U_values),len(self.J_values)))
		self.Final_params = np.zeros(self.Initial_params.shape)
		self.Convergence_Grid = np.zeros((len(self.U_values),len(self.J_values)))

	def Phase_Diagram_point(self,v):
		Model = self.Model
		Sol = self.Solver

		Model.U, Model.J = self.U_values[v[0]],self.J_values[v[1]]
		Model.MF_params = self.Initial_params[v]

		Sol.Itterate(verbose=False)

		Model.E_occ = Sol.total_occupied_energy

		Model.Calculate_Energy()

		if Sol.converged == False:
			Model.Final_Total_Energy = np.inf

		if self.verbose == True:
			if Sol.converged ==True:
				print('U:',round(Model.U,2),'J:', round(Model.J,2),'Initial MFP:',self.Initial_params[v], 'Final MFP:',Model.MF_params)
			else:
				print('U:',round(Model.U,2),'J:', round(Model.J,2),'Initial MFP:',self.Initial_params[v], 'Did Not Converge')

		return Model.Final_Total_Energy, Model.MF_params, Sol.converged

	def Sweep(self, outfolder, Final_Run=False):
		# MP way
		PD_grid = itertools.product(self.U_idx,self.J_idx)
		with Pool(self.n_threads) as p:
			results = p.map(self.Phase_Diagram_point, PD_grid)

		# Energies results list to array to csv
		PD_grid = itertools.product(self.U_idx,self.J_idx)
		for i,v in enumerate(PD_grid):
			self.Es_trial[v] = results[i][0]
			self.Final_params[v] = results[i][1]
			self.Convergence_Grid[v] = results[i][2]

		self.Convergence_Grid = self.Convergence_Grid.astype(int)
		self.Convergence_pc = 100*np.mean(self.Convergence_Grid)

		if Final_Run == False:
			np.savetxt(os.path.join(outfolder,'Energies.csv'),self.Es_trial,delimiter=',')
			# np.savetxt(os.path.join(outfolder,'Convergence_Grid.csv'),self.Convergence_Grid,delimiter=',')
			# for i in range(self.Initial_params.shape[2]):
				# outfile = os.path.join(outfolder,'MF_Solutions','MF'+str(i)+'.csv')
				# np.savetxt(outfile,self.Final_params[:,:,i],delimiter=",")

		if Final_Run == True:
			np.savetxt(os.path.join(outfolder,'Energies.csv'),self.Es_trial,delimiter=',')
			np.savetxt(os.path.join(outfolder,'Convergence_Grid.csv'),self.Convergence_Grid,delimiter=',')
			for i in range(self.Initial_params.shape[2]):
				outfile = os.path.join(outfolder,'MF_Solutions','MF'+str(i)+'.csv')
				np.savetxt(outfile,self.Final_params[:,:,i],delimiter=",")