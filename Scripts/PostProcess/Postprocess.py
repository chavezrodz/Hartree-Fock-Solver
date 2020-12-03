import numpy as np
import os
from Code.Nickelates.Hamiltonian import Hamiltonian
from Code.Display.ResultsPlots import sweeper_plots

Model_Params = dict(
    N_shape=(50, 50),
    Filling=0.25,
    BZ_rot=1,
    stress=0,
    Delta_CT=0,
    eps=0,
    t_1=1,
    t_2=0.15,
    t_4=0.05,
    U=1,
    J=1)

i, j = 'U', 'J',
i_values = np.linspace(0, 4, 30)
j_values = np.linspace(0, 0.5, 30)

method = 'sigmoid'
beta = 1.5
Itteration_limit = 250
tolerance = 1e-3
bw_norm = True

Model = Hamiltonian(Model_Params)

run_folders = 'Meta_4'
for folder in sorted(os.listdir(os.path.join('Results', run_folders)))[300:]:
    frf = os.path.join('Results', run_folders, folder, 'Final_Results')
    sweeper_plots(i, i_values, j, j_values, Model.Dict, frf, BW_norm=bw_norm)
    print(frf)
    # break
    # break
    # guesses_folder = os.path.join('Results', run_folders, folder, 'Guesses_Results')
    # for guess in os.listdir(guesses_folder):
    #     guess = os.path.join(guesses_folder, guess)
    #     print(guess)
    #     sweeper_plots(i, i_values, j, j_values, Model.Dict, guess, BW_norm=True)
