import sys
import numpy as np
import os
import shutil
from time import time
from models.Nickelates.Hamiltonian import Hamiltonian
from solver.HFA_Solver import HFA_Solver
import solver.calculations as calc
from solver.PhaseDiagramSweeper import Phase_Diagram_Sweeper
from solver.Optimizer import Optimizer_exhaustive as Optimizer_exhaustive
import utils as u


def trial_iteration(Model, Solver, Sweeper, MF_params,
                    outfolder, Include_MFPs, logging):
    os.makedirs(outfolder, exist_ok=True)
    if logging:
        sys.stdout = open(outfolder+'/logs.txt', 'w+')
    t0 = time()
    Sweeper.Sweep()
    Sweeper.save_results(outfolder, Include_MFPs)
    dt = time() - t0
    return dt, Sweeper.Convergence_pc


def find_fermi_bw(model_params, sweeper_args, solver_args):
    Model = Hamiltonian(model_params)
    Solver = HFA_Solver(Model, **solver_args)
    setattr(Model, sweeper_args['variables'][0], 0)
    setattr(Model, sweeper_args['variables'][1], 0)
    setattr(Model, 'eps', 0)
    setattr(Model, 'Delta_CT', 0)
    Solver.Iterate(verbose=False)
    calc.bandwidth(Model)
    fermi_bw = Model.fermi_bw
    print(f'Fermi_bw: {fermi_bw}')
    return fermi_bw


def generate_diagram(batch_folder, model_params, params_list, sweeper_args,
                     solver_args, bw_norm=True,
                     guess_run=True, final_run=True, rm_guesses=True,
                     logging=True):

    Run_ID = u.make_id(sweeper_args, model_params)
    Results_Folder = os.path.join('Results', batch_folder, Run_ID)
    os.makedirs(Results_Folder, exist_ok=True)
    standard = sys.stdout

    if bw_norm:
        fermi_bw = find_fermi_bw(model_params, sweeper_args, solver_args)
        unnormed_x = model_params['eps']
        unnormed_y = model_params['Delta_CT']

        model_params['eps'] = model_params['eps'] * fermi_bw
        model_params['Delta_CT'] = model_params['Delta_CT'] * fermi_bw

        normed_x = model_params['eps']
        normed_y = model_params['Delta_CT']
        norming_settings = {
            'fermi_bw': fermi_bw,
            'unnormed_x': unnormed_x,
            'normed_x': normed_x,
            'unnormed_y': unnormed_y,
            'normed_y': normed_y,
            }
    else:
        norming_settings = {}

    u.write_settings(
        Run_ID, Results_Folder,
        [model_params, solver_args, sweeper_args, norming_settings]
        )

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
            Sweeper = Phase_Diagram_Sweeper(
                Model, Solver, MF_params, **sweeper_args, fermi_bw=fermi_bw
                )

            dt, convergence_pc = trial_iteration(
                Model, Solver, Sweeper, MF_params, outfolder,
                Include_MFPs=sweeper_args['save_guess_mfps'], logging=logging)

            print(f'Diagram iteration: {n} time to complete (s): {round(dt,3)} Converged points:{round(convergence_pc,3)} % \n')

    if final_run:
        a = time()
        Input_Folder = os.path.join(Results_Folder, 'Guesses_Results')
        Final_Results_Folder = os.path.join(Results_Folder, 'Final_Results')

        os.makedirs(Final_Results_Folder, exist_ok=True)
        if logging:
            sys.stdout = open(Final_Results_Folder+'/logs.txt', 'w+')

        Model = Hamiltonian(model_params)
        Solver = HFA_Solver(Model, **solver_args)

        optimal_guesses, Optimal_Energy = Optimizer_exhaustive(
            Input_Folder, params_list,
            input_MFP=sweeper_args['save_guess_mfps'],
            verbose=sweeper_args['verbose'])

        Sweeper = Phase_Diagram_Sweeper(
            Model, Solver, optimal_guesses, **sweeper_args, fermi_bw=fermi_bw
            )

        Sweeper.Sweep()
        Sweeper.save_results(Final_Results_Folder, Include_MFPs=True)

        Final_Energies = Sweeper.Es_trial
        close_check = np.allclose(Optimal_Energy, Final_Energies, atol=1e-2)
        print(f'Guesses and final calculations consistent:{close_check}')
        print(f'time to complete (s):{round(time()-a,3)} Converged points:{round(Sweeper.Convergence_pc,3)} % \n')

        if rm_guesses:
            shutil.rmtree(os.path.join(Results_Folder, 'Guesses_Results'))

    sys.stdout = standard
