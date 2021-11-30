import numpy as np
import os
import itertools


def tuplelist(x):
    LIST = []
    for v in itertools.product(*x):
        LIST.append(v)
    return LIST


def make_id(sweeper_args, model_params):
    Run_ID = 'Iterated:'
    Run_ID = Run_ID + '_'.join("{!s}".format(key)
                               for (key) in sweeper_args['variables'])
    Run_ID = Run_ID + '_'
    Run_ID = Run_ID + '_'.join("{!s}={!r}".format(key, val)
                               for (key, val) in model_params.items())
    return Run_ID


def write_settings(Run_ID, Results_Folder, dicts):
    settings = open(Results_Folder+"/settings.txt", "w+")
    settings.write('Run_ID:' + Run_ID + '\n')
    for dic in dicts:
        for (key, val) in dic.items():
            settings.write("{!s}={!r} \n".format(key, val))
    settings.close()


def Read_MFPs(folder):
    N = len(os.listdir(folder))
    for i in range(N):
        file = os.path.join(folder, 'MF'+str(i)+'.csv')
        if i == 0:
            MF = np.loadtxt(file, delimiter=',')
        else:
            MF = np.dstack((MF, np.loadtxt(file, delimiter=',')))
    return MF


def load_energies_conv(input_folder, folderlist):
    # Stack all energies,convergence arrays
    E_Tower, C_Tower = [], []
    print("Loading Energies")
    for i, folder in enumerate(folderlist):
        print('\t', folder)

        E_file = os.path.join(input_folder, folder, 'Energies.csv')
        C_file = os.path.join(input_folder, folder, 'Convergence.csv')

        E_Tower.append(np.loadtxt(E_file, delimiter=','))
        C_Tower.append(np.loadtxt(C_file, delimiter=','))

    E_Tower = np.stack(E_Tower, axis=-1)
    C_Tower = np.stack(C_Tower, axis=-1).astype(bool)
    return E_Tower, C_Tower


def load_solutions(input_folder, folderlist):
    # Recover best solutions from all guesses
    print('Loading Solutions')
    Solutions = []
    for i, folder in enumerate(folderlist):
        print('\t', folder)
        MFPs = Read_MFPs(os.path.join(input_folder, folder, 'MF_Solutions'))
        Solutions.append(MFPs)

    Solutions = np.stack(Solutions, axis=-1)
    Solutions = np.swapaxes(Solutions, -1, -2)
    return Solutions
