import numpy as np
import os
import matplotlib.pyplot as plt
import itertools
import Code.Utils as U
import Code.Nickelates.Interpreter as In


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
        Model_Params[meta_args['y_label']] = meta_args['y_values'][closest_y_ind]

        Run_ID = U.make_id(sweeper_args, Model_Params)
        mfps = U.Read_MFPs(os.path.join('Results', Batch_Folder, Run_ID, 'Final_Results', 'MF_Solutions'))
        value_at_origin = Diagram_stats(mfps, meta_args['tracked_state'])

        for x, y in itertools.product(np.arange(len(x_values)), np.arange(len(y_values))):
            Model_Params[meta_args['x_label']] = x_values[x]
            Model_Params[meta_args['y_label']] = y_values[y]
            Run_ID = U.make_id(sweeper_args, Model_Params)
            print(Run_ID)
            sol_path = os.path.join('Results', Batch_Folder, Run_ID, 'Final_Results', 'MF_Solutions')
            mfps = U.Read_MFPs(sol_path)
            MetaArray[x, y] = Diagram_stats(mfps, meta_args['tracked_state'])
        print('Loading Done')
        MetaArray -= value_at_origin
        np.savetxt(os.path.join('Results', Batch_Folder, 'MetaArray.csv'), MetaArray, delimiter=',')

    return MetaArray


def Diagram_stats(mfps, phase):
    Phases = In.array_interpreter(mfps)[:, :, 1:]
    # Phases = In.arr_to_int(Phases)
    Size = np.size(Phases)
    Uniques, counts = np.unique(Phases, return_counts=True)
    counts = counts/Size * 100
    phase_ind = np.where(Uniques == phase)
    if len(*phase_ind) == 0:
        return 0
    else:
        return counts[phase_ind]


def make_meta_fig(MetaArray, meta_args, font=14):
    f, ax = plt.subplots(figsize=(8, 5))
    # ax.set_xlabel(meta_args['x_label'])
    # ax.set_ylabel(meta_args['y_label'])

    ax.set_xlabel(r'$\epsilon_b, [t_1]$', fontsize=font)
    ax.set_ylabel(r'$\Delta_{CF}, [t_1]$', fontsize=font)

    ax.set(frame_on=False)
    # ax.set_title(r'Relative occupancy of $\uparrow \downarrow,  \bar{z} \bar{z}$')

    CS = ax.contour(MetaArray.T, colors='red', levels=[0])
    ax.clabel(CS, inline=True, fontsize=font, fmt='% 1.1f')

    ax.plot([0., len(meta_args['x_values'])], [0, len(meta_args['y_values'])], c='black')

    CM = ax.pcolormesh(MetaArray.T, cmap='RdBu', vmin=-np.max(np.abs(MetaArray)), vmax=np.max(np.abs(MetaArray)))
    # plt.colorbar(CM)
    cbar = plt.colorbar(CM)
    cbar.ax.set_ylabel(r'% change in $\uparrow \downarrow, \bar{z} \bar{z}$ phase area', fontsize=font)

    plt.tick_params(axis='both', which='major', labelsize=font)
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
    plt.savefig('MetaDiagram.png', bbox_inches='tight')
    plt.show()
