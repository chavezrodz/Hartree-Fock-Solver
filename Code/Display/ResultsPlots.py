import numpy.linalg as LA
import sys
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
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
        plt.xlabel(i_label)
        plt.ylabel(j_label)

        positions = np.linspace(0, len(i_values), 4)
        ticks = np.round(np.linspace(min(i_values), max(i_values), 4),2)
        plt.xticks(positions, ticks)

        positions = np.linspace(0, len(j_values), 4)
        ticks = np.round(np.linspace(min(j_values), max(j_values), 4),2)
        plt.yticks(positions, ticks)

        plt.colorbar()

        if results_folder is not None:
            MFPs_folder = os.path.join(results_folder, 'Plots', 'Mean Field Parameters')
            os.makedirs(MFPs_folder, exist_ok=True)
            plt.savefig(MFPs_folder+'/'+Dict[i]+'.png', transparent=transparent)
        if show:
            plt.show()
        plt.close()


def feature_plot(feature, i_label, i_values, j_label, j_values, results_folder, show, transparent):
    plt.title(feature)
    plt.pcolormesh(np.loadtxt(results_folder+'/'+feature+'.csv', delimiter=',').T)
    plt.xlabel(i_label)
    plt.ylabel(j_label)

    positions = np.linspace(0, len(i_values), 4)
    ticks = np.round(np.linspace(min(i_values), max(i_values), 4),2)
    plt.xticks(positions, ticks)

    positions = np.linspace(0, len(j_values), 4)
    ticks = np.round(np.linspace(min(j_values), max(j_values), 4),2)
    plt.yticks(positions, ticks)

    plt.colorbar()

    if results_folder is not None:
        features_folder = os.path.join(results_folder,'Plots','Features')
        os.makedirs(features_folder, exist_ok=True)
        feature_file = os.path.join(features_folder, feature+'.png')
        plt.savefig(feature_file, transparent=transparent)
    if show:
        plt.show()
    plt.close()


def phases_plot(Phase, i_label, i_values, j_label, j_values, results_folder, show, transparent):
    CM = Phase[:, :, 0]
    spin_orb = Phase[:, :, 1]
    unique_states = np.unique(spin_orb).astype(int)

    f, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlabel(i_label)
    ax.set_ylabel(j_label)
    ax.set(frame_on=False)

    # ticks
    N_x = np.min([len(i_values), 5])
    N_y = np.min([len(j_values), 5])
    plt.xticks(np.linspace(0, len(i_values), N_x), np.linspace(np.min(i_values), np.max(i_values), N_x))
    plt.yticks(np.linspace(0, len(j_values), N_y), np.linspace(np.min(j_values), np.max(j_values), N_y))

    # Charge Contour
    CS = ax.contour(np.abs(CM.T), colors='red', levels=[0.1, 0.3, 0.5])
    ax.clabel(CS, inline=True, fontsize=10)

    # spin-orbit
    cmap = plt.cm.get_cmap('prism', In.N_possible_states)
    im = ax.pcolormesh(spin_orb.T, alpha=1, cmap=cmap, vmin=0, vmax=In.N_possible_states-1)
    patches = [mpatches.Patch(color=cmap(state), label=In.pos_to_label[state]) for state in unique_states]
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, prop={"size": 13})
    plt.tight_layout()

    if results_folder is not None:
        plt.savefig(results_folder+'/Plots/PhaseDiagram.png', transparent=transparent)
    if show:
        plt.show()
    plt.close()


def one_dimensional_phases(Phase, i_label, i_values, results_folder, show, transparent):
    CM = Phase[:, :, 0]
    spin_orb = Phase[:, :, 1]
    unique_states = np.unique(spin_orb).astype(int)

    f, ax = plt.subplots(figsize=(8, 2))
    ax.set_title('1d GS phase')
    ax.set_xlabel(i_label)
    ax.set(frame_on=False)
    positions = np.linspace(0, len(i_values), 4)
    ticks = np.round(np.linspace(min(i_values), max(i_values), 4),2)
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
    x = np.atleast_2d(np.transpose(i_values))[0, :]
    x = x + np.mean(np.diff(x))/2
    x = x[:-1]
    # print(x.shape)

    for i in range(len(features)):
        label = features[i]
        array = (arrays[i])
        if array.ndim == 1:
            array = np.expand_dims(array, -1)
        y = np.diff(array, axis=0)
        y = LA.norm(y, axis=-1)
        plt.plot(x, y, label=label)

    plt.xlabel(i_label)
    plt.yscale('log')
    plt.tight_layout()
    plt.legend()

    if results_folder is not None:
        plt.savefig(results_folder +'/Plots/multi_differences.png', transparent=transparent)
    if show:
        plt.show()
    plt.close()


def sweeper_plots(i_label, i_values, j_label, j_values, Dict, final_results_folder=None, show=False, transparent=False, BW_norm=False):
    Solutions_folder = os.path.join(final_results_folder,'MF_Solutions')
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
