import Code.Utils as Utils
import numpy as np
import os
from Code.Nickelates.Hamiltonian import Hamiltonian
from Code.Display.ResultsPlots import sweeper_plots

Model_Params = dict(
    N_shape=(5, 5),
    Filling=0.25,
    BZ_rot=1,
    stress=-1,
    Delta_CT=0,
    eps=0,
    t_1=1,
    t_2=0.15,
    t_4=0.05,
    U=1,
    J=1)

i, j = 'U', 'J',
i_values = np.linspace(0, 3, 10)
j_values = np.linspace(0, 6, 10)

method = 'sigmoid'
beta = 1.5
Itteration_limit = 250
tolerance = 1e-3
bw_norm = False

verbose = True
save_guess_mfps = True

epsilons = [0, 1]
delta_cts = [0, 1]

model_params_lists = Utils.tuplelist([epsilons, delta_cts])

for (Model_Params['eps'], Model_Params['Delta_CT']) in model_params_lists:

    Run_ID = 'Itterated:'+str(i)+'_'+str(j)+'_'
    Run_ID = Run_ID + '_'.join("{!s}={!r}".format(key, val) for (key, val) in Model_Params.items())
    mfps = Utils.Read_MFPs(os.path.join('meta',Run_ID,'Final_Results','MF_Solutions'))
    print(Run_ID)

# Model = Hamiltonian(Model_Params)

# run_folders = 'tent/tent'
# for folder in os.listdir(os.path.join('Results', run_folders)):
#     frf = os.path.join('Results', run_folders, folder, 'Final_Results')
#     print(frf)
#     sweeper_plots(i, i_values, j, j_values, Model.Dict, frf, BW_norm=bw_norm)
#     # break
#     # break
#     # guesses_folder = os.path.join('Results', run_folders, folder, 'Guesses_Results')
#     # for guess in os.listdir(guesses_folder):
#     #     guess = os.path.join(guesses_folder, guess)
#     #     print(guess)
#     #     sweeper_plots(i, i_values, j, j_values, Model.Dict, guess, BW_norm=True)
