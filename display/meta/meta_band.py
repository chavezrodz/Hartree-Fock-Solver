import itertools
import numpy as np
import os
import utils as utils


def meta_band(input_folder, trials_list):
    """
    In progress: Generate meta-bandstructure
    """
    n_mfps = len(trials_list[0])
    folderlist = ['Guess'+str(MF_params)
                  for MF_params in np.array(trials_list)]
    E_Tower, C_Tower = utils.load_energies_conv(input_folder, folderlist)
    diag_shape = E_Tower.shape[:-1]

    solutions = utils.load_solutions(input_folder, folderlist)
    unconverged_sols = np.empty(solutions.shape)
    unconverged_sols[:] = np.nan

    print(solutions.shape)

    return

