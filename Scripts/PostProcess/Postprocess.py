import numpy as np
import os
from Code.Nickelates.Hamiltonian import Hamiltonian
from Code.Display.ResultsPlots import sweeper_plots

Model_Params = dict(
    N_shape=(10, 10),
    eps=0,
    Delta_CT=0
    )

# Diagram Ranges
i, j = 'U', 'J'
i_values = np.linspace(0, 1, 10)
j_values = np.linspace(0, 0.25, 10)
method = 'sigmoid'
beta = 1.5
Itteration_limit = 250
tolerance = 1e-3
bw_norm = True

verbose = True
save_guess_mfps = False

Batch_Folder = 'Dummy'
Model = Hamiltonian(Model_Params)

for folder in sorted(os.listdir(os.path.join('Results', Batch_Folder))):
    frf = os.path.join('Results', Batch_Folder, folder, 'Final_Results')
    print(frf)
    sweeper_plots(i, i_values, j, j_values, Model.Dict, frf, BW_norm=bw_norm, show=False)
    # break
    # break
    # guesses_folder = os.path.join('Results', run_folders, folder, 'Guesses_Results')
    # for guess in os.listdir(guesses_folder):
    #     guess = os.path.join(guesses_folder, guess)
    #     print(guess)
    #     sweeper_plots(i, i_values, j, j_values, Model.Dict, guess, BW_norm=True)
