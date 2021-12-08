import matplotlib.patches as mpatches
import os
from itertools import product as product
import numpy as np
import argparse
import scripts.script_diagrams as diagrams
from display.meta.meta_band import meta_band
import utils
import models.Nickelates.Interpreter as In
import matplotlib.pyplot as plt

# Command Line Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_threads', type=int, default=8)
args = parser.parse_args()


batch_folder = 'diagrams_test'
logging = False
rm_guesses = False
bw_norm = True

sweeper_args = dict(
    variables=['U', 'J'],
    values_list=[np.linspace(0, 1, 30),
                 np.linspace(0, 0.2, 30)],
    bw_norm=bw_norm,
    save_guess_mfps=True,
    verbose=True,
    n_threads=args.n_threads
    )

solver_args = dict(
    method='sigmoid',
    beta=1.5,
    Iteration_limit=250,
    tol=1e-3,
    )


trials_list = np.array([
    [0., 0, 0, 0., 0.],
    [0.8, 1.0, 0.0, 0.7, 0.15],
    [1.0, 0.6, 0.0, 0.7, 0.15],
    [0.0, 0.2, 0.5, 0.0, 0.2],
    [0.2, 0.5, 1.0, 1.0, 0.0],
    [0.5, 0.5, 0.0, 0.5, 0.1],
    [1., 1., 1., 1, 1.]

])

model_params = {
    'eps': 0,
    'Delta_CT': 0,
    'k_res': 128
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
    e_tower, c_tower = utils.load_energies_conv(input_folder, folderlist)
    diag_shape = e_tower.shape[:-1]

    solutions = utils.load_solutions(input_folder, folderlist)
    unconverged_sols = np.empty(solutions.shape)
    unconverged_sols[:] = np.nan

    cut_idx = 1
    e_cut = e_tower[:, :]
    sol_cut = solutions[:, :]
    conv_cut = c_tower[:, :]

    sols_int = In.array_interpreter(sol_cut)[..., 1]
    # print(sol_cut.shape)
    # print(sols_interpreted)
    print(sols_int.shape)
    print(e_cut.shape)
    print(conv_cut)
    unique_states = np.unique(sols_int).astype(int)

    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    for x in np.arange(e_cut.shape[0]):
        for y in np.arange(e_cut.shape[1]):
            for j in np.arange(e_cut.shape[-1]):
                state = int(sols_int[x, y, j])
                ax.scatter(
                    x, y,  e_cut[x, y, j],
                    '.',
                    color=In.col_dict[state],
                    label=In.pos_to_label[state],
                    alpha=0.5
                 )

    patches = [
        mpatches.Patch(color=In.col_dict[state], label=In.pos_to_label[state])
        for state in unique_states
        ]

    plt.legend(handles=patches)
    plt.show()

    return


# meta_band(input_folder, trials_list)

diagrams.generate_diagram(
    batch_folder, model_params,
    trials_list, sweeper_args, solver_args,
    bw_norm=bw_norm, logging=logging, rm_guesses=rm_guesses
    )
