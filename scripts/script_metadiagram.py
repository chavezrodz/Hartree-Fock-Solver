import numpy as np
import os
import itertools
import utils as U
import models.Nickelates.Interpreter as In
from display.meta.metadiagram import make_meta_fig


def Diagram_stats(mfps, phase):
    Phases = In.array_interpreter(mfps)[:, :, 1:]
    Uniques, counts = np.unique(Phases, return_counts=True)
    phase_ind = np.where(Uniques == phase)
    if len(*phase_ind) == 0:
        return 0
    else:
        return counts[phase_ind]


def generate_meta_array(Model_Params, sweeper_args, meta_args):
    x_values = meta_args['x_values']
    y_values = meta_args['y_values']
    Batch_Folder = meta_args['Batch_Folder']
    results_folder = meta_args['results_folder']

    meta_shape = (len(meta_args['x_values']), len(meta_args['y_values']))
    MetaArray = np.zeros(meta_shape)

    print('Loading Results')
    # Finding value at origin
    closest_x_ind = np.argmin(np.abs(meta_args['x_values']))
    closest_y_ind = np.argmin(np.abs(meta_args['y_values']))

    Model_Params[meta_args['x_label']] = meta_args['x_values'][closest_x_ind]
    Model_Params[meta_args['y_label']] = meta_args['y_values'][closest_y_ind]

    Run_ID = U.make_id(sweeper_args, Model_Params)
    mfps = U.Read_MFPs(
        os.path.join(results_folder, Batch_Folder, Run_ID, 'Final_Results', 'MF_Solutions')
        )
    value_at_origin = Diagram_stats(mfps, meta_args['tracked_state'])

    for x, y in itertools.product(np.arange(len(x_values)), np.arange(len(y_values))):
        Model_Params[meta_args['x_label']] = x_values[x]
        Model_Params[meta_args['y_label']] = y_values[y]
        Run_ID = U.make_id(sweeper_args, Model_Params)
        print(Run_ID)
        sol_path = os.path.join(results_folder, Batch_Folder, Run_ID, 'Final_Results', 'MF_Solutions')
        mfps = U.Read_MFPs(sol_path)
        MetaArray[x, y] = Diagram_stats(mfps, meta_args['tracked_state'])
    print('Loading Done')
    MetaArray = (MetaArray - value_at_origin)/value_at_origin * 100
    np.savetxt(os.path.join(results_folder, Batch_Folder, 'MetaArray.csv'), MetaArray, delimiter=',')

    return MetaArray


def make_meta_diag(Model_Params, sweeper_args, meta_args):
    Batch_Folder = meta_args['Batch_Folder']
    results_folder = meta_args['results_folder']

    if meta_args['load']:
        meta_arr = np.loadtxt(
            os.path.join(results_folder, Batch_Folder, 'MetaArray.csv'),
            delimiter=',')
    else:
        meta_arr = generate_meta_array(Model_Params, sweeper_args, meta_args)

    make_meta_fig(meta_arr, meta_args)
