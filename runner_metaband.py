import os
from itertools import product as product
import numpy as np
import argparse
import scripts.script_diagrams as diagrams
from display.diagrams.diag_analysis import convergence_per_trials, plot_metaband, plot_e_cut_derivative, find_phase
import utils
import models.Nickelates.Interpreter as In


batch_folder = 'diagrams_test'
bw_norm = True

sweeper_args = dict(
    variables=['U', 'J'],
    values_list=[np.linspace(0, 1, 30),
                 np.linspace(0, 0.2, 30)],
    bw_norm=bw_norm,
    save_guess_mfps=True,
    verbose=True
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

x_fixed_idx = 0
fixed_value = 0.5

input_folder = utils.make_id(sweeper_args, model_params)
input_folder = os.path.join('Results', batch_folder, input_folder)
out_folder = os.path.join(input_folder, 'analysis')
os.makedirs(out_folder, exist_ok=True)

n_mfps = len(trials_list[0])
folderlist = ['Guess'+str(MF_params)
              for MF_params in np.array(trials_list)]
guess_folder = os.path.join(input_folder, 'Guesses_Results')
e_tower, c_tower = utils.load_energies_conv(guess_folder, folderlist)
solutions = utils.load_solutions(guess_folder, folderlist)
sols_int = In.array_interpreter(solutions)[..., 1]
unique_states = np.unique(sols_int).astype(int)
convergence_per_trials(c_tower, out_folder)


# metaband
phases = [find_phase(phase_idx, sols_int, e_tower, c_tower)
          for phase_idx in unique_states]
colors = [In.col_dict[phase_idx] for phase_idx in unique_states]
labels = [In.pos_to_label[state] for state in unique_states]
plot_metaband(phases, colors, labels,
              sweeper_args['variables'], sweeper_args['values_list'],
              out_folder)

# Energy cut
x_fixed = sweeper_args['variables'][x_fixed_idx]
fixed_value_idx = np.argmin(np.abs(
    sweeper_args['values_list'][x_fixed_idx] - fixed_value
    ))


plot_e_cut_derivative(phases, colors, labels,
                      x_fixed_idx, fixed_value_idx,
                      sweeper_args, out_folder)
