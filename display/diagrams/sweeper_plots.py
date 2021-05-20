import sys
import utils as utils
import os
import Code.Nickelates.Interpreter as In
import seaborn as sns

sns.set_theme()
sns.set_context("paper")


def sweeper_plots(i_label, i_values, j_label, j_values, Dict, final_results_folder=None, show=False, transparent=False, BW_norm=False):
    Solutions_folder = os.path.join(final_results_folder, 'MF_Solutions')
    if not os.path.exists(Solutions_folder): print('Solutions not found'); sys.exit(2)

    Plots_folder = os.path.join(final_results_folder, 'Plots')
    os.makedirs(Plots_folder, exist_ok=True)

    if BW_norm:
        i_label = i_label+'/W'; j_label = j_label+'/W'

    MF = utils.Read_MFPs(Solutions_folder)
    MFP_plots(MF, i_label, i_values, j_label, j_values, Dict, final_results_folder, show, transparent)

    Phase = In.array_interpreter(MF)
    phases_plot(Phase, i_label, i_values, j_label, j_values, final_results_folder)

    features = ['Energies', 'Distortion', 'Convergence', 'Conductance']
    for feature in features:
        feature_plot(feature, i_label, i_values, j_label, j_values, final_results_folder, show, transparent)
