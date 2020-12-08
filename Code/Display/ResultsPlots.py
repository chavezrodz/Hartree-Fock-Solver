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


def MFP_plots(MFPs, i_label, i_values, j_label, j_values, Dict, results_folder, show, transparent):
    for i in range(len(Dict)):
        arr = np.abs(MFPs[:, :, i].T)
        plt.pcolormesh(arr, vmin=0, vmax=1)
        plt.title(Dict[i])
        plt.xlabel(i_label)
        plt.ylabel(j_label)
        plt.xticks(np.linspace(0, len(i_values), 4), np.linspace(0, max(i_values), 4, dtype=int))
        plt.yticks(np.linspace(0, len(j_values), 4), np.linspace(0, max(j_values), 4, dtype=int))
        plt.colorbar()
        if results_folder is not None:
            MFPs_folder = os.path.join(results_folder, 'Plots', 'Mean Field Parameters')
            if not os.path.exists(MFPs_folder): os.mkdir(MFPs_folder)
            plt.savefig(MFPs_folder+'/'+Dict[i]+'.png', transparent=transparent)
        if show:
            plt.show()
        plt.close()


def feature_plot(feature, i_label, i_values, j_label, j_values, results_folder, show, transparent):
    plt.title(feature)
    plt.pcolormesh(np.loadtxt(results_folder+'/'+feature+'.csv', delimiter=',').T)
    plt.xlabel(i_label)
    plt.ylabel(j_label)
    plt.xticks(np.linspace(0, len(i_values), 4), np.linspace(0, max(i_values), 4, dtype=int))
    plt.yticks(np.linspace(0, len(j_values), 4), np.linspace(0, max(j_values), 4, dtype=int))
    plt.colorbar()

    if results_folder is not None:
        features_folder = os.path.join(results_folder,'Plots','Features')
        if not os.path.exists(features_folder): os.mkdir(features_folder)
        plt.savefig(features_folder+'/'+feature+'.png', transparent=transparent)
    if show:
        plt.show()
    plt.close()


def phases_plot(Phase,i_label, i_values, j_label,j_values, results_folder, show, transparent):
    CM = Phase[:, :, 0]
    MF_Spin_orb = Phase[:, :, 1:]
    spin_orb = In.arr_to_int(MF_Spin_orb)
    unique_states = np.unique(spin_orb)

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
    cmap = plt.cm.get_cmap('prism', 120)
    im = ax.pcolormesh(spin_orb.T, alpha=1, cmap=cmap, vmin=0, vmax=119)
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
    MF_Spin_orb = Phase[:, :, 1:]
    spin_orb = In.arr_to_int(MF_Spin_orb)
    unique_states = np.unique(spin_orb)

    f, ax = plt.subplots(figsize=(8, 2))
    ax.set_title('1d cut')
    ax.set_xlabel(i_label)
    ax.set(frame_on=False)
    plt.xticks(np.linspace(0, len(i_values), 4), np.linspace(0, max(i_values), 4, dtype=int))
    ax.set_yticks([])
    # Charge Contour
    # print(CM)

    # spin-orbit
    cmap = plt.cm.get_cmap('prism', 120)
    im = ax.pcolormesh(spin_orb, alpha=1, cmap=cmap, vmin=0, vmax=119)
    patches = [mpatches.Patch(color=cmap(state), label=In.pos_to_label[state]) for state in unique_states]
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, prop={"size": 13})

    plt.tight_layout()
    if results_folder is not None:
        plt.savefig(results_folder+'/Plots/1Dcut.png', transparent=transparent)
    if show:
        plt.show()
    plt.close()


def density_of_states(i_label, i_value, Energies, Fermi_Energy, results_folder, show, transparent):
    plt.hist(Energies.flatten(), bins='fd')
    plt.title('Density of states')
    plt.axvline(Fermi_Energy, label='Fermi Energy', color='red')
    plt.xlabel('Energy (Ev)')
    plt.legend()
    plt.savefig(results_folder+'/DOS/'+i_label+str(i_value)+'.png', transparent=transparent)
    plt.close()


def E_Plots(i_label, i_values, Dict, guesses, final_results_folder=None, show=False, transparent=False):
    j_label = 'Guesses'
    j_values = np.arange(len(guesses))

    Plots_folder = os.path.join(final_results_folder, 'Plots')
    if not os.path.exists(Plots_folder): os.mkdir(Plots_folder)

    # All guesses states
    Solutions_folder = os.path.join(final_results_folder,'MF_Solutions')
    if not os.path.exists(Solutions_folder): print('Solutions not found'); sys.exit(2)
    MF = Utils.Read_MFPs(Solutions_folder)

    MFP_plots(MF, i_label, i_values, j_label, j_values, Dict, final_results_folder, show, transparent)

    Phase = In.array_interpreter(MF)
    phases_plot(Phase,i_label, i_values, j_label, j_values, final_results_folder, show, transparent)

    # Ground state phases
    GS_folder = os.path.join(final_results_folder, 'GS_Solutions')
    if not os.path.exists(GS_folder): print('Solutions not found'); sys.exit(2)
    GS_MF = Utils.Read_MFPs(GS_folder)

    Phase = In.array_interpreter(GS_MF)
    one_dimensional_phases(Phase, i_label, i_values, final_results_folder, show, transparent)

    # Density of states
    DOS_folder = os.path.join(final_results_folder,'DOS')
    if not os.path.exists(DOS_folder): os.mkdir(DOS_folder)

    sol_energies = np.loadtxt(os.path.join(final_results_folder,'Solution_Energies.csv'), delimiter=',')
    fermis = np.loadtxt(os.path.join(final_results_folder,'Fermi_Energies.csv'),delimiter=',')
    for i, v in enumerate(i_values):
        Energies = sol_energies[i]
        Fermi_Energy = fermis[i]
        density_of_states(i_label, v, Energies, Fermi_Energy, final_results_folder, show, transparent)

    features = ['Energies', 'Distortion', 'Convergence', 'Conductance']
    for feature in features:
        feature_plot(feature, i_label, i_values, j_label, j_values, final_results_folder, show, transparent)


def sweeper_plots(i_label, i_values, j_label, j_values, Dict, final_results_folder=None, show=False, transparent=False, BW_norm=False):
    Solutions_folder = os.path.join(final_results_folder,'MF_Solutions')
    if not os.path.exists(Solutions_folder): print('Solutions not found'); sys.exit(2)

    Plots_folder = os.path.join(final_results_folder,'Plots') 
    if not os.path.exists(Plots_folder): os.mkdir(Plots_folder)

    if BW_norm: i_label = i_label+'/W'; j_label = j_label+'/W'

    MF = Utils.Read_MFPs(Solutions_folder)
    MFP_plots(MF, i_label, i_values, j_label, j_values, Dict, final_results_folder, show, transparent)

    Phase = In.array_interpreter(MF)
    phases_plot(Phase, i_label, i_values, j_label, j_values, final_results_folder, show, transparent)

    features = ['Energies', 'Distortion','Convergence','Conductance']
    for feature in features:
        feature_plot(feature,i_label, i_values, j_label, j_values, final_results_folder, show, transparent)
