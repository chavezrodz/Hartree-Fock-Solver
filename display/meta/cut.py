import numpy.linalg as LA
import sys
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import utils as Utils
import numpy as np
import os
import models.Nickelates.Interpreter as In

import seaborn as sns

# sns.set_theme()
# sns.set_context("paper")


def MFP_plots(MFPs, i_label, i_values, j_label, j_values, Dict, results_folder, show, transparent, standardize=False):
    for i in range(len(Dict)):
        arr = MFPs[:, :, i].T
        if standardize:
            arr = np.abs(arr)
            plt.pcolormesh(arr, vmin=0, vmax=1)
        else:
            plt.pcolormesh(arr)
        plt.title(Dict[i])
        plt.xlabel(i_label, fontsize=16)
        plt.ylabel(j_label, fontsize=16)

        positions = np.linspace(0, len(i_values), 4)
        ticks = np.round(np.linspace(min(i_values), max(i_values), 4),2)
        plt.xticks(positions, ticks)

        positions = np.linspace(0, len(j_values), 4)
        ticks = np.round(np.linspace(min(j_values), max(j_values), 4),2)
        plt.yticks(positions, ticks)

        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.colorbar()

        plt.tight_layout()

        if results_folder is not None:
            MFPs_folder = os.path.join(results_folder, 'Plots', 'Mean Field Parameters')
            os.makedirs(MFPs_folder, exist_ok=True)
            plt.savefig(MFPs_folder+'/'+Dict[i]+'.png', transparent=transparent)
        if show:
            plt.show()
        plt.close()


def feature_plot(feature, i_label, i_values, j_label, j_values, results_folder, show, transparent):
    # plt.title(feature)
    plt.pcolormesh(np.loadtxt(results_folder+'/'+feature+'.csv', delimiter=',').T)

    plt.xlabel(r'$'+i_label+'$', fontsize=16)
    plt.ylabel(r'$'+j_label+'$', fontsize=16)

    positions = np.linspace(0, len(i_values), 4)
    ticks = np.round(np.linspace(min(i_values), max(i_values), 4), 2)
    plt.xticks(positions, ticks)

    positions = np.linspace(0, len(j_values), 4)
    ticks = np.round(np.linspace(min(j_values), max(j_values), 4), 2)
    plt.yticks(positions, ticks)

    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.colorbar()

    plt.tight_layout()

    if results_folder is not None:
        features_folder = os.path.join(results_folder, 'Plots', 'Features')
        os.makedirs(features_folder, exist_ok=True)
        feature_file = os.path.join(features_folder, feature+'.png')
        plt.savefig(feature_file, transparent=transparent)
    if show:
        plt.show()
    plt.close()


def phases_plot(Phase, i_label, i_values, j_label, j_values, results_folder, font=20):
    CM = Phase[:, :, 0]
    spin_orb = Phase[:, :, 1]
    OS = Phase[:, :, 2]
    unique_states = np.unique(spin_orb).astype(int)

    # print(unique_states)
    # unique_labels = [In.pos_to_label[state] for state in unique_states]
    # for label in unique_labels:
    #     print(label)

    f, ax = plt.subplots(figsize=(8, 8))  # or 6,5 without legend
    # ax.set_xlabel(i_label)
    # ax.set_ylabel(j_label)
    ax.set_xlabel(r'$'+i_label+'$', fontsize=font)
    ax.xaxis.set_label_coords(0.5, -0.02)

    ax.set_ylabel(r'$'+j_label+'$', fontsize=font)
    ax.yaxis.set_label_coords(-0.02, 0.5)

    ax.set(frame_on=False)

    # ticks
    # N_x = np.min([len(i_values), 5])
    # N_y = np.min([len(j_values), 5])
    # plt.xticks(np.linspace(0, len(i_values), N_x), np.linspace(np.min(i_values), np.max(i_values), N_x))
    # plt.yticks(np.linspace(0, len(j_values), N_y), np.linspace(np.min(j_values), np.max(j_values), N_y))

    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels([0, 0.25, '', 0.75, 1])
    ax.xaxis.set_ticks_position('bottom')

    ax.set_yticks([0, 0.05, 0.1, 0.15, 0.2])
    ax.set_yticklabels([0, 0.05, '', 0.15, 0.2])
    ax.yaxis.set_ticks_position('left')

    plt.tick_params(axis='both', which='major', labelsize=20)

    countour_label_font = 20

    # Charge Contour
    CS = ax.contour(np.abs(CM.T), colors='red', levels=[0.01, 0.1, 0.3, 0.5],
                    linewidths=2, extent=(0, 1, 0, 0.2))
    ax.clabel(CS, inline=True, fontsize=countour_label_font, fmt='% 1.1f')

    # Orbital Contour
    OS = ax.contour(np.abs(OS.T), colors='purple', levels=[0.01, 0.1, 0.5, 0.9],
                    linestyles='dashed', linewidths=2, extent=(0, 1, 0, 0.2))

    ax.clabel(OS, inline=True, fontsize=countour_label_font, fmt='% 1.1f')
    ax.grid(linewidth=0)

    # spin-orbit
    cmap = In.custom_cmap
    norm = In.custom_norm
    col_dict = In.col_dict

    ax.imshow(np.rot90(spin_orb), cmap=cmap, norm=norm, aspect='auto', extent=(0, 1, 0, 0.2))

    patches = [mpatches.Patch(color=col_dict[state], label=In.pos_to_label[state]) for state in unique_states]

    # Plot Specific marker points
    Metallic = {'x': 0.2, 'y': 0.1,
                'label': 'Metallic',
                'marker': 'o', 'color': 'black'}

    AFM = {'x': 0.8, 'y': 0.05,
           'label': 'AFM',
           'marker': '*', 'color': 'black'}

    FM = {'x': 0.8, 'y': 0.1,
          'label': 'FM',
          'marker': 's', 'color': 'black'}

    CD = {'x': 0.5, 'y': 0.2,
          'label': 'CD',
          'marker': 'd', 'color': 'black'}

    points = [Metallic, AFM, FM, CD]

    for point in points:
        ax.plot(point['x'], point['y'],
                marker=point['marker'], color=point['color'],
                markersize=10, label=point['label']
                )

        # patches.append(mlines.Line2D([], [], linestyle='None',
        #                marker=point['marker'], color=point['color'],
        #                markersize=10, label=point['label'])
        #                )

    ax.legend(handles=patches,
              bbox_to_anchor=(0.5, -0.075),
              loc=9, borderaxespad=0.0,
              prop={"size": font}, fontsize=font,
              ncol=3, labelspacing=0.2, columnspacing=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, 'Plots', 'PhaseDiagram.png'))
    plt.close()


def one_dimensional_phases(Phase, i_label, i_values, results_folder, show, transparent):
    CM = Phase[:, :, 0]
    spin_orb = Phase[:, :, 1]

    # CM = Phase[:, :, 0]
    # OS = Phase[:, :, 1]
    # spin_orb = Phase[:, :, 2]

    unique_states = np.unique(spin_orb).astype(int)

    f, ax = plt.subplots(figsize=(8, 2))
    ax.set_title('1d GS phase')
    ax.set_xlabel(i_label)
    ax.set(frame_on=False)
    positions = np.linspace(0, len(i_values), 4)
    ticks = np.round(np.linspace(min(i_values), max(i_values), 4), 2)
    plt.xticks(positions, ticks)
    ax.set_yticks([])
    # Charge Contour
    # print(CM)

    # spin-orbit
    cmap = plt.cm.get_cmap('prism', In.N_possible_states)
    im = ax.pcolormesh(spin_orb, alpha=1, cmap=cmap, vmin=0, vmax=In.N_possible_states-1)
    patches = [mpatches.Patch(color=cmap(state), label=In.pos_to_label[state]) for state in unique_states]
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, prop={"size": 13})

    plt.tight_layout()
    if results_folder is not None:
        plt.savefig(results_folder+'/Plots/1D_GS_phase.png', transparent=transparent)
    if show:
        plt.show()
    plt.close()


def one_d_feature(array, feature, i_label, i_values, results_folder=None,
                  show=False, transparent=False):
    plt.plot(i_values, array)
    plt.xlabel(i_label)
    plt.ylabel(feature)
    plt.tight_layout()
    if results_folder is not None:
        plt.savefig(results_folder +'/Plots/'+ feature +'.png', transparent=transparent)
    if show:
        plt.show()
    plt.close()


def difference_plots(features, arrays, i_label, i_values,
                     results_folder=None, log_scaled=False):
    x = i_values
    x = x + np.mean(np.diff(x))/2
    x = x[:-1]

    fig, ax = plt.subplots()
    for i in range(len(features)):
        label = features[i]
        array = (arrays[i])
        if array.ndim == 1:
            array = np.expand_dims(array, -1)
        y = np.diff(array, axis=0)
        y = LA.norm(y, axis=-1)

        if i == 0:
            ax.plot(x, y, label=label, color='black', linestyle='-')
        elif i == 1:
            ax.plot(x, y, label=label, color='blue', linestyle='--')
        else:
            ax.plot(x, y, label=label)
    # plt.xlabel(i_label)
    plt.xlabel('N', fontsize=16)
    plt.ylabel('Error in Mean-Field Parameters (Dimensionless)\n'+r'and Energy ($t_1^{0}$)', fontsize=16)

    if log_scaled:
        plt.yscale('log')

    plt.tick_params(axis='both', which='both', labelsize=14)
    plt.legend(prop={"size": 14})
    ax.set_facecolor("white")
    ax.grid(b=True, color='grey', linewidth=0.3)
    for spine in ax.spines.values():
        spine.set_color('0.3')


    plt.tight_layout()
    if results_folder is not None:
        plt.savefig(results_folder + '/Plots/multi_differences.png', bbox_inches='tight')
    plt.show()
    plt.close()


def one_d_plots(i_label, i_values, Dict, guesses, final_results_folder=None, show=False, transparent=False):
    j_label = 'Guesses'
    j_values = np.arange(len(guesses))

    Plots_folder = os.path.join(final_results_folder, 'Plots')
    os.makedirs(Plots_folder, exist_ok=True)

    # All guesses states
    Solutions_folder = os.path.join(final_results_folder, 'MF_Solutions')
    if not os.path.exists(Solutions_folder): print('Solutions not found'); sys.exit(2)
    MF = Utils.Read_MFPs(Solutions_folder)

    MFP_plots(MF, i_label, i_values, j_label, j_values, Dict, final_results_folder, show, transparent)

    Phase = In.array_interpreter(MF)
    phases_plot(Phase, i_label, i_values, j_label, j_values, final_results_folder)

    # GS Energies
    sol_energies = np.loadtxt(os.path.join(final_results_folder, 'GS_Energy.csv'), delimiter=',')
    one_d_feature(sol_energies, 'GS Energy ', i_label, i_values, final_results_folder, show, transparent)
    # Ground state phases
    GS_MF = np.loadtxt(os.path.join(final_results_folder, 'GS_Solutions.csv'), delimiter=',')
    difference_plots(
        ['d|MFP| (Dimensionless)', r'd|E| ($t_1^{0}$)'], [GS_MF, sol_energies],
        i_label, i_values, final_results_folder)

    Phase = In.array_interpreter(np.expand_dims(GS_MF, axis=0))
    one_dimensional_phases(Phase, i_label, i_values, final_results_folder, show, transparent)

    features = ['Energies', 'Distortion', 'Convergence', 'Conductance']
    for feature in features:
        feature_plot(feature, i_label, i_values, j_label, j_values, final_results_folder, show, transparent)
