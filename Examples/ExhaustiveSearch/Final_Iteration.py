import Code.Utils as Utils
import numpy as np
import argparse
import Code.Scripts.diagrams as diagrams
import sys
# Command Line Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_threads', type=int, default=8)
parser.add_argument('--run_ind', type=int, default=0)
args = parser.parse_args()

Batch_Folder = 'Exhaustive'
logging = True

sweeper_args = dict(
    variables=['U', 'J'],
    values_list=[np.linspace(0, 1, 10),
                 np.linspace(0, 0.25, 10)],
    bw_norm=True,
    save_guess_mfps=False,
    verbose=True,
    n_threads=args.n_threads
    )

solver_args = dict(
    method='sigmoid',
    beta=1.5,
    Itteration_limit=150,
    tolerance=1e-3,
    tol=1e-3,
    )

model_params = dict(
    N_shape=(100, 100),
    Delta_CT=0,
    eps=0)

deltas = np.linspace(0, 1, 3)
sfm = np.linspace(0, 1, 3)
Deltas_FO = np.linspace(0, 1, 3)
safm = np.linspace(0, 1, 3)
Deltas_AFO = np.linspace(0, 1, 3)

all_params = Utils.tuplelist([deltas, sfm, Deltas_FO, safm, Deltas_AFO])

params_list = all_params

print('Diagram itterations starting')

diagrams.generate_diagram(Batch_Folder, model_params, params_list,
                          sweeper_args, solver_args, guess_run=False,
                          final_run=True, logging=logging, rm_guesses=False)

sys.stdout = open("/dev/stdout", "w")
print('Finished succesfully')
