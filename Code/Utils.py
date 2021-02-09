import numpy as np
import os
import itertools


def tuplelist(x):
    LIST = []
    for v in itertools.product(*x):
        LIST.append(v)
    return LIST


def Read_MFPs(folder):
    N = len(os.listdir(folder))
    for i in range(N):
        file = os.path.join(folder, 'MF'+str(i)+'.csv')
        if i == 0:
            MF = np.loadtxt(file, delimiter=',')
        else:
            MF = np.dstack((MF, np.loadtxt(file, delimiter=',')))
    return MF


def make_id(sweeper_args, model_params):
    Run_ID = 'Iterated:'
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