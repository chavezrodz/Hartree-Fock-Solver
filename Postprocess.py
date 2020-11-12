import numpy as np
import sys
import os
import Code.Utils as Utils
from Code.Nickelates.Hamiltonian import Hamiltonian
from Code.Display.ResultsPlots import sweeper_plots
i,j = 'U','J',
i_values = np.linspace(0,6,35)
j_values = np.linspace(0,3,35)

Model_Params = dict(
N_shape = (50,50),
Filling = 0.25,
BZ_rot = 1,
stress=0,
eps = 0,
t_1 = 1,
t_2 = 0.15,
t_4 = 0.05,
U = 1,
J = 1)
Model = Hamiltonian(Model_Params)

run_folders = 'Tests'
for folder in os.listdir(os.path.join('Results',run_folders)):
	frf = os.path.join('Results',run_folders,folder,'Final_Results')
	print(frf)
	sweeper_plots(i,i_values,j,j_values,Model.Dict,frf, BW_norm=True)
	guesses_folder = os.path.join('Results',run_folders,folder,'Guesses_Results')
	for guess in os.listdir(guesses_folder):
		guess = os.path.join(guesses_folder,guess)
		print(guess)
		sweeper_plots(i,i_values,j,j_values,Model.Dict,guess, BW_norm=True)