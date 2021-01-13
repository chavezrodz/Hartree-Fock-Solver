import sys
import numpy as np
import os
import shutil
from time import time
import matplotlib.pyplot as plt
import itertools
import Code.Utils as Utils
import Code.Nickelates.Interpreter as In
from Code.Nickelates.Hamiltonian import Hamiltonian
from Code.Solver.HFA_Solver import HFA_Solver
from Code.Solver.PhaseDiagramSweeper import Phase_Diagram_Sweeper
from Code.Solver.Optimizer import Optimizer_exhaustive as Optimizer_exhaustive


def make_id(sweeper_args, model_params):
    Run_ID = 'Itterated:'
    Run_ID = Run_ID + '_'.join("{!s}".format(key)
                               for (key) in sweeper_args['variables'])
    Run_ID = Run_ID + '_'
    Run_ID = Run_ID + '_'.join("{!s}={!r}".format(key, val)
                               for (key, val) in model_params.items())
    return Run_ID


def write_settings(Run_ID, Results_Folder, model_params, solver_args, sweeper_args):
    settings = open(Results_Folder+"/settings.txt", "w+")
    settings.write('Run_ID:' + Run_ID + '\n')
    for (key, val) in model_params.items():
        settings.write("{!s}={!r} \n".format(key, val))
    for (key, val) in solver_args.items():
        settings.write("{!s}={!r} \n".format(key, val))
    for (key, val) in sweeper_args.items():
        settings.write("{!s}={!r} \n".format(key, val))
    settings.close()


def trial_itteration(Model, Solver, Sweeper, MF_params, outfolder,
                     Include_MFPs, logging):
    os.makedirs(outfolder, exist_ok=True)
    if logging:
        sys.stdout = open(outfolder+'/logs.txt', 'w+')
    t0 = time()
    Sweeper.Sweep()
    Sweeper.save_results(outfolder, Include_MFPs)
    dt = time() - t0
    return dt, Sweeper.Convergence_pc


def generate_diagram(batch_folder, model_params, params_list, sweeper_args,
                     solver_args, guess_run=True, final_run=True, rm_guesses=True, logging=True):
    Run_ID = make_id(sweeper_args, model_params)
    Results_Folder = os.path.join('Results', batch_folder, Run_ID)
    os.makedirs(Results_Folder, exist_ok=True)
    standard = sys.stdout
    write_settings(Run_ID, Results_Folder, model_params, solver_args, sweeper_args)

    if guess_run:
        for n in range(len(params_list)):
            # Guesses Input

            MF_params = np.array(params_list[n])
            Guess_Name = 'Guess'+str(MF_params)
            outfolder = os.path.join(Results_Folder, 'Guesses_Results', Guess_Name)
            os.makedirs(outfolder, exist_ok=True)
            if logging:
                sys.stdout = open(outfolder+'/logs.txt', 'w+')

            Model = Hamiltonian(model_params, MF_params)
            Solver = HFA_Solver(Model, **solver_args)

            Sweeper = Phase_Diagram_Sweeper(Model, Solver, MF_params, **sweeper_args)

            dt, convergence_pc = trial_itteration(Model, Solver, Sweeper, MF_params,
                                                  outfolder, Include_MFPs=sweeper_args['save_guess_mfps'], logging=logging)

            print(f'Diagram itteration: {n} time to complete (s): {round(dt,3)} Converged points:{round(convergence_pc,3)} % \n')

    if final_run:
        a = time()
        Input_Folder = os.path.join(Results_Folder, 'Guesses_Results')
        Final_Results_Folder = os.path.join(Results_Folder, 'Final_Results')

        os.makedirs(Final_Results_Folder, exist_ok=True)
        if logging:
            sys.stdout = open(Final_Results_Folder+'/logs.txt', 'w+')

        Model = Hamiltonian(model_params)
        Solver = HFA_Solver(Model, **solver_args)

        Optimal_guesses, Optimal_Energy = Optimizer_exhaustive(Input_Folder, params_list, input_MFP=sweeper_args['save_guess_mfps'])

        sweeper = Phase_Diagram_Sweeper(Model, Solver, Optimal_guesses, **sweeper_args)

        sweeper.Sweep()
        sweeper.save_results(Final_Results_Folder, Include_MFPs=True)

        Final_Energies = sweeper.Es_trial
        print(f'Initial guess sweep and final calculations are consistent:{np.array_equal(Final_Energies, Optimal_Energy)}')
        print(f'time to complete (s):{round(time()-a,3)} Converged points:{round(sweeper.Convergence_pc,3)} % \n')

        if rm_guesses:
            shutil.rmtree(os.path.join(Results_Folder, 'Guesses_Results'))

    sys.stdout = standard


def get_meta_array(Model_Params, sweeper_args, meta_args):
    x_values = meta_args['x_values']
    y_values = meta_args['y_values']
    Batch_Folder = meta_args['Batch_Folder']

    if meta_args['load']:
        MetaArray = np.loadtxt(os.path.join('Results', meta_args['Batch_Folder'], 'MetaArray.csv'), delimiter=',')
    else:
        meta_shape = (len(meta_args['x_values']), len(meta_args['y_values']))
        MetaArray = np.zeros(meta_shape)

        print('Loading Results')
        # Finding value at origin
        closest_x_ind = np.argmin(np.abs(meta_args['x_values']))
        closest_y_ind = np.argmin(np.abs(meta_args['y_values']))

        Model_Params[meta_args['x_label']] = meta_args['x_values'][closest_x_ind]
        Model_Params[meta_args['y_label']] = meta_args['x_values'][closest_y_ind]

        Run_ID = make_id(sweeper_args, Model_Params)
        mfps = Utils.Read_MFPs(os.path.join('Results', Batch_Folder, Run_ID, 'Final_Results', 'MF_Solutions'))
        value_at_origin = Diagram_stats(mfps, meta_args['tracked_state'])
        for x, y in itertools.product(np.arange(len(x_values)), np.arange(len(y_values))):
            Model_Params[meta_args['x_label']] = x_values[x]
            Model_Params[meta_args['y_label']] = y_values[y]
            Run_ID = make_id(sweeper_args, Model_Params)
            print(Run_ID)
            sol_path = os.path.join('Results', Batch_Folder, Run_ID, 'Final_Results', 'MF_Solutions')
            mfps = Utils.Read_MFPs(sol_path)
            MetaArray[x, y] = Diagram_stats(mfps, meta_args['tracked_state'])
        print('Loading Done')
        MetaArray -= value_at_origin
        np.savetxt(os.path.join('Results', Batch_Folder, 'MetaArray.csv'), MetaArray, delimiter=',')

    return MetaArray


def Diagram_stats(mfps, phase):
    Phases = In.array_interpreter(mfps)[:, :, 1:]
    Phases = In.arr_to_int(Phases)
    Size = np.size(Phases)
    Uniques, counts = np.unique(Phases, return_counts=True)
    counts = counts/Size * 100
    phase_ind = np.where(Uniques == phase)
    if len(*phase_ind) == 0:
        return 0
    else:
        return counts[phase_ind]


def make_meta_fig(MetaArray, meta_args):
    f, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlabel(meta_args['x_label'])
    ax.set_ylabel(meta_args['y_label'])
    ax.set(frame_on=False)
    ax.set_title(r'Relative occupancy of $\uparrow \downarrow,  \bar{z} \bar{z}$')

    CS = ax.contour(MetaArray.T, colors='red', levels=[0])
    ax.clabel(CS, inline=True, fontsize=10)

    ax.plot([0., len(meta_args['x_values'])], [0, len(meta_args['y_values'])], c='black')

    CM = ax.pcolormesh(MetaArray.T, cmap='RdBu', vmin=-np.max(np.abs(MetaArray)), vmax=np.max(np.abs(MetaArray)))
    plt.colorbar(CM)

    x_values = meta_args['x_values']
    y_values = meta_args['y_values']
    N_x = np.min([len(x_values), 5])
    N_y = np.min([len(y_values), 5])
    plt.xticks(np.linspace(0, len(x_values), N_x), np.linspace(np.min(x_values), np.max(x_values), N_x))
    plt.yticks(np.linspace(0, len(y_values), N_y), np.linspace(np.min(y_values), np.max(y_values), N_y))

    # N_x = 4
    # N_y = 4
    # plt.xticks(np.linspace(0, len(meta_args['x_values']), N_x),  [0, 0.25, 0.5, 1])
    # plt.yticks(np.linspace(0, len(meta_args['y_values']), N_y), [0, 0.25, 0.5, 1])

    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig('metadiag.png')
    plt.show()