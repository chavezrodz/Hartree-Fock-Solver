import shutil
import numpy as np
import os
from Code.Nickelates.Hamiltonian import Hamiltonian
from Code.Display.two_d_map import make_2d_map
from Code.Display.ResultsPlots import sweeper_plots

i, j = 'U', 'J'
i_values = np.linspace(0, 1, 30)
j_values = np.linspace(0, 0.25, 30)
bw_norm = True
Model = Hamiltonian()

make_map = True
process_guesses = False
full_analysis = False

Batch_Folders = ['strain_zero', 'strain_tensile', 'strain_compressive']
Results_Folder = 'Final_Results'

for Batch_Folder in Batch_Folders:
    if make_map:
        make_2d_map(Results_Folder, Batch_Folder)
    folder_list = sorted(os.listdir(os.path.join(Results_Folder, Batch_Folder)))
    folder_list = folder_list[:1]

    for folder in folder_list:
        frf = os.path.join(Results_Folder, Batch_Folder, folder, 'Final_Results')
        if os.path.exists(frf):
            print('Processing: ', folder)
            if full_analysis:
                sweeper_plots(i, i_values, j, j_values, Model.Dict,
                              frf, BW_norm=bw_norm, show=False)

            if process_guesses:
                guesses_folder = os.path.join(
                    'Results', Batch_Folder, folder, 'Guesses_Results')
                for guess in os.listdir(guesses_folder):
                    guess = os.path.join(guesses_folder, guess)
                    print('Processing guess:', guess)
                    sweeper_plots(i, i_values, j, j_values, Model.Dict,
                                  guess, BW_norm=True)

    if full_analysis:
        Diagrams_Folder = os.path.join(Results_Folder, Batch_Folder+'_diags')
        os.makedirs(Diagrams_Folder, exist_ok=True)

        for folder in folder_list:
            frf = os.path.join(Results_Folder, Batch_Folder, folder, 'Final_Results')
            if os.path.exists(frf):
                diagram = os.path.join(frf, 'Plots', 'PhaseDiagram.png')
                out = os.path.join(Diagrams_Folder, folder+'.png')
                shutil.copy(diagram, out)
