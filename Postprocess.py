import shutil
import numpy as np
import os
from Code.Nickelates.Hamiltonian import Hamiltonian
from Code.Display.ResultsPlots import sweeper_plots


Results_Folder = 'Final_Results'
process_guesses = False
i, j = 'U', 'J'
i_values = np.linspace(0, 1, 30)
j_values = np.linspace(0, 0.25, 30)
bw_norm = True

for Batch_Folder in ['strain_zero', 'strain_tensile', 'strain_compressive']:
    Model = Hamiltonian()
    for folder in sorted(os.listdir(os.path.join(Results_Folder, Batch_Folder))):
        frf = os.path.join(Results_Folder, Batch_Folder, folder, 'Final_Results')
        if os.path.exists(frf):
            print('Processing: ', folder)
            sweeper_plots(i, i_values, j, j_values, Model.Dict,
                          frf, BW_norm=bw_norm, show=False)

            if process_guesses:
                guesses_folder = os.path.join(
                    'Results', Batch_Folder, folder, 'Guesses_Results')
                for guess in os.listdir(guesses_folder):
                    guess = os.path.join(guesses_folder, guess)
                    print(guess)
                    sweeper_plots(i, i_values, j, j_values, Model.Dict,
                                  guess, BW_norm=True)

    Diagrams_Folder = os.path.join(Results_Folder, Batch_Folder, Batch_Folder)
    os.makedirs(Diagrams_Folder, exist_ok=True)

    for folder in sorted(os.listdir(os.path.join(Results_Folder, Batch_Folder))):
        frf = os.path.join(Results_Folder, Batch_Folder, folder, 'Final_Results')
        if os.path.exists(frf):
            diagram = os.path.join(frf, 'Plots', 'PhaseDiagram.png')
            out = os.path.join(Diagrams_Folder, folder+'.png')
            shutil.copy(diagram, out)
