import numpy as np
import os
from Code.Nickelates.Hamiltonian import Hamiltonian
from Code.Display.ResultsPlots import sweeper_plots


Batch_Folder = 'Test'

i, j = 'U', 'J',
i_values = np.linspace(0, 1, 10)
j_values = np.linspace(0, 0.1, 10)
bw_norm = True

Model = Hamiltonian()

for folder in sorted(os.listdir(os.path.join('Results', Batch_Folder))):
    frf = os.path.join('Results', Batch_Folder, folder, 'Final_Results')
    print(frf)
    sweeper_plots(i, i_values, j, j_values, Model.Dict, frf, BW_norm=bw_norm, show=False)
    # break
    # guesses_folder = os.path.join('Results', Batch_Folder, folder, 'Guesses_Results')
    # for guess in os.listdir(guesses_folder):
    #     guess = os.path.join(guesses_folder, guess)
    #     print(guess)
    #     sweeper_plots(i, i_values, j, j_values, Model.Dict, guess, BW_norm=True)
