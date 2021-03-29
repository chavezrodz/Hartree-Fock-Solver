import numpy.linalg as LA
import sys
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from matplotlib.colors import BoundaryNorm
import Code.Utils as Utils
import numpy as np
import os
import Code.Nickelates.Interpreter as In

import seaborn as sns

sns.set_theme()
sns.set_context("paper")


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

    plt.xlabel(r'$'+i_label+', [t_1]$', fontsize=16)
    plt.ylabel(r'$'+j_label+', [t_1]$', fontsize=16)

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


def phases_plot(Phase, i_label, i_values, j_label, j_values, results_folder, show, transparent):
    CM = Phase[:, :, 0]
    spin_orb = Phase[:, :, 1]
    OS = Phase[:, :, 2]
    unique_states = np.unique(spin_orb).astype(int)

    # print(unique_states)
    # unique_labels = [In.pos_to_label[state] for state in unique_states]
    # for label in unique_labels:
    #     print(label)

    f, ax = plt.subplots(figsize=(8, 5))  # or 6,5 without legend
    # ax.set_xlabel(i_label)
    # ax.set_ylabel(j_label)
    ax.set_xlabel(r'$'+i_label+', [t_1]$', fontsize=20)
    ax.set_ylabel(r'$'+j_label+', [t_1]$', fontsize=20)

    ax.set(frame_on=False)

    # ticks
    # N_x = np.min([len(i_values), 5])
    # N_y = np.min([len(j_values), 5])
    # plt.xticks(np.linspace(0, len(i_values), N_x), np.linspace(np.min(i_values), np.max(i_values), N_x))
    # plt.yticks(np.linspace(0, len(j_values), N_y), np.linspace(np.min(j_values), np.max(j_values), N_y))

    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels([0, '', 0.5, '', 1])
    ax.xaxis.set_ticks_position('bottom')

    ax.set_yticks([0, 0.05, 0.1, 0.15, 0.2, 0.25])
    ax.set_yticklabels([0, '', 0.1, '', 0.2, 0.25])
    ax.yaxis.set_ticks_position('left')

    plt.tick_params(axis='both', which='major', labelsize=16)

    # Charge Contour
    CS = ax.contour(np.abs(CM.T), colors='red', levels=[0.1, 0.3, 0.5],
                    linewidths=2, extent=(0, 1, 0, 0.25))
    ax.clabel(CS, inline=True, fontsize=14)

    OS = ax.contour(np.abs(OS.T), colors='purple', levels=[0.1, 0.5],
                    linestyles='dashed', linewidths=2, extent=(0, 1, 0, 0.25))
    ax.clabel(OS, inline=True, fontsize=14)
    ax.grid(linewidth=0)

    # spin-orbit
    cmap = In.custom_cmap
    norm = In.custom_norm
    col_dict = In.col_dict

    ax.imshow(np.rot90(spin_orb), cmap=cmap, norm=norm, aspect='auto', extent=(0, 1, 0, 0.25))

    patches = [mpatches.Patch(color=col_dict[state], label=In.pos_to_label[state]) for state in unique_states]
    ax.legend(handles=patches, bbox_to_anchor=(1.1, 1), loc=2, borderaxespad=0.0, prop={"size": 13})
    plt.tight_layout()

    if results_folder is not None:
        plt.savefig(results_folder+'/Plots/PhaseDiagram.png', transparent=transparent)
    if show:
        plt.show()
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
                     results_folder=None, show=False, transparent=False):
    x = i_values
    x = x + np.mean(np.diff(x))/2
    x = x[:-1]

    for i in range(len(features)):
        label = features[i]
        array = (arrays[i])
        if array.ndim == 1:
            array = np.expand_dims(array, -1)
        y = np.diff(array, axis=0)
        y = LA.norm(y, axis=-1)
        plt.plot(x, y, label=label)

    # plt.xlabel(i_label)
    plt.xlabel('Momentum Resolution', fontsize=16)
    # plt.yscale('log')
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.legend(prop={"size": 13})

    if results_folder is not None:
        plt.savefig(results_folder +'/Plots/multi_differences.png', transparent=transparent)
    if show:
        plt.show()
    plt.close()


def sweeper_plots(i_label, i_values, j_label, j_values, Dict, final_results_folder=None, show=False, transparent=False, BW_norm=False):
    Solutions_folder = os.path.join(final_results_folder, 'MF_Solutions')
    if not os.path.exists(Solutions_folder): print('Solutions not found'); sys.exit(2)

    Plots_folder = os.path.join(final_results_folder, 'Plots')
    os.makedirs(Plots_folder, exist_ok=True)

    if BW_norm: i_label = i_label+'/W'; j_label = j_label+'/W'

    MF = Utils.Read_MFPs(Solutions_folder)
    MFP_plots(MF, i_label, i_values, j_label, j_values, Dict, final_results_folder, show, transparent)

    Phase = In.array_interpreter(MF)
    phases_plot(Phase, i_label, i_values, j_label, j_values, final_results_folder, show, transparent)

    features = ['Energies', 'Distortion', 'Convergence', 'Conductance']
    for feature in features:
        feature_plot(feature, i_label, i_values, j_label, j_values, final_results_folder, show, transparent)


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
    phases_plot(Phase, i_label, i_values, j_label, j_values, final_results_folder, show, transparent)

    # GS Energies
    sol_energies = np.loadtxt(os.path.join(final_results_folder, 'GS_Energy.csv'), delimiter=',')
    one_d_feature(sol_energies, 'GS Energy ', i_label, i_values, final_results_folder, show, transparent)
    # Ground state phases
    GS_MF = np.loadtxt(os.path.join(final_results_folder, 'GS_Solutions.csv'), delimiter=',')
    difference_plots(
        ['d|MFP|', 'd|E|'], [GS_MF, sol_energies],
        i_label, i_values, final_results_folder, show, transparent)

    Phase = In.array_interpreter(np.expand_dims(GS_MF, axis=0))
    one_dimensional_phases(Phase, i_label, i_values, final_results_folder, show, transparent)

    features = ['Energies', 'Distortion', 'Convergence', 'Conductance']
    for feature in features:
        feature_plot(feature, i_label, i_values, j_label, j_values, final_results_folder, show, transparent)
