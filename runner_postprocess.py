import numpy as np
import os
from models.Nickelates.Hamiltonian import Hamiltonian
from scripts.script_postprocess import postprocess


if __name__ == '__main__':
    results_folder = 'Final_Results'

    i, j = 'U', 'J'
    i_values = np.linspace(0, 1, 30)
    j_values = np.linspace(0, 0.25, 30)

    make_map = False
    full = True

    mfp_dict = Hamiltonian().Dict

    Batch_Folders = [
        'strain_zero',
        # 'strain_tensile',
        # 'strain_compressive'
        ]

    for batch in Batch_Folders:
        if batch == 'strain_zero':
            bw_norm = r'$W$'
        elif batch == 'strain_compressive':
            bw_norm = r'$W_{c}$'
        elif batch == 'strain_tensile':
            bw_norm = r'$W_{t}$'

        folder_list = sorted(os.listdir(os.path.join(results_folder, batch)))
        postprocess(
            results_folder, batch, folder_list,
            i, i_values, j, j_values,
            mfp_dict, bw_norm,
            full=full, make_map=make_map)
