import utils as utils
import os
import models.Nickelates.Interpreter as In
from display.diagrams.mfps_plots import mfps_plots
from display.diagrams.phase_diagram import phases_diagram
from display.diagrams.feature_plot import feature_plot
import seaborn as sns

sns.set_theme()
sns.set_context("paper")


def sweeper_plots(i_label, i_values, j_label, j_values, mfp_dict,
                  final_results_folder=None, show=False,
                  transparent=False, BW_norm=None):

    Solutions_folder = os.path.join(final_results_folder, 'MF_Solutions')

    if os.path.exists(Solutions_folder):
        pass
    else:
        raise Exception('Solutions not found')

    Plots_folder = os.path.join(final_results_folder, 'Plots')
    os.makedirs(Plots_folder, exist_ok=True)

    if BW_norm is not None:
        i_label = i_label + '/' + BW_norm
        j_label = j_label + '/' + BW_norm

    MF = utils.Read_MFPs(Solutions_folder)
    mfps_plots(MF, i_label, i_values, j_label, j_values, mfp_dict,
               final_results_folder, show, transparent)

    Phase = In.array_interpreter(MF)
    phases_diagram(Phase, i_label, i_values, j_label, j_values,
                   final_results_folder)

    features = ['Energies', 'Distortion', 'Convergence', 'Conductance']

    for feature in features:
        feature_plot(feature, i_label, i_values, j_label, j_values,
                     final_results_folder, show, transparent)
