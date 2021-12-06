import os
from itertools import product as product
import numpy as np
import argparse
import scripts.script_diagrams as diagrams
from display.meta.meta_band import meta_band
import utils

import models.Nickelates.Interpreter as In


# Command Line Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_threads', type=int, default=8)
args = parser.parse_args()

batch_folder = 'diagrams_test'
rm_guesses = False
bw_norm = True
sweeper_args = dict(
    variables=['U', 'J'],
    values_list=[np.linspace(0, 1, 3),
                 np.linspace(0, 0.2, 3)],
    bw_norm=bw_norm,
    save_guess_mfps=True,
    verbose=True,
    n_threads=args.n_threads
    )

trials_list = np.array([
    [0.8, 1.0, 0.0, 0.7, 0.15],
    [1.0, 0.6, 0.0, 0.7, 0.15],
    [0.0, 0.2, 0.5, 0.0, 0.2],
    [0.2, 0.5, 1.0, 1.0, 0.0],
    [0.5, 0.5, 0.0, 0.5, 0.1]
])

model_params = {
    'eps': 0,
    'Delta_CT': 0,
    'k_res': 32
    }

input_folder = utils.make_id(sweeper_args, model_params)
input_folder = os.path.join('Results', batch_folder, input_folder, 'Guesses_Results')


def meta_band(input_folder, trials_list):
    """
    In progress: Generate meta-bandstructure
    """
    n_mfps = len(trials_list[0])
    folderlist = ['Guess'+str(MF_params)
                  for MF_params in np.array(trials_list)]
    e_Tower, c_Tower = utils.load_energies_conv(input_folder, folderlist)
    diag_shape = e_Tower.shape[:-1]

    solutions = utils.load_solutions(input_folder, folderlist)
    unconverged_sols = np.empty(solutions.shape)
    unconverged_sols[:] = np.nan

    cut_idx = 0

    sol_cut = solutions[cut_idx]
    conv_cut = c_Tower[cut_idx]

    print(conv_cut)
    print(sol_cut.shape)

    # sols_interpreted = 


    return


meta_band(input_folder, trials_list)

# diagrams.generate_diagram(
#     batch_folder, model_params,
#     trials_list, sweeper_args, solver_args,
#     bw_norm=bw_norm, logging=logging, rm_guesses=rm_guesses
#     )


