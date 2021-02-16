import sys
import numpy as np
import os
import shutil
from time import time
from Code.Nickelates.Hamiltonian import Hamiltonian
from Code.Solver.HFA_Solver import HFA_Solver
from Code.Solver.PhaseDiagramSweeper import Phase_Diagram_Sweeper
from Code.Solver.Optimizer import Optimizer_exhaustive as Optimizer_exhaustive
import Code.Utils as U


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


def generate_diagram(batch_folder, model_params, params_list, sweeper_args,
                     solver_args, guess_run=True, final_run=True, rm_guesses=True, logging=True):
    Run_ID = U.make_id(sweeper_args, model_params)
    Results_Folder = os.path.join('Results', batch_folder, Run_ID)
    os.makedirs(Results_Folder, exist_ok=True)
    standard = sys.stdout
    U.write_settings(Run_ID, Results_Folder, model_params, solver_args, sweeper_args)

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

        Optimal_guesses, Optimal_Energy = Optimizer_exhaustive(
            Input_Folder, params_list, input_MFP=sweeper_args['save_guess_mfps'])

        sweeper = Phase_Diagram_Sweeper(Model, Solver, Optimal_guesses, **sweeper_args)

        sweeper.Sweep()
        sweeper.save_results(Final_Results_Folder, Include_MFPs=True)

        Final_Energies = sweeper.Es_trial
        print(np.abs(Final_Energies - Optimal_Energy))
        print(f'Initial guess sweep and final calculations are consistent:{np.allclose(Final_Energies, Optimal_Energy, atol=1e-2)}')
        print(f'time to complete (s):{round(time()-a,3)} Converged points:{round(sweeper.Convergence_pc,3)} % \n')

        if rm_guesses:
            shutil.rmtree(os.path.join(Results_Folder, 'Guesses_Results'))

    sys.stdout = standard
