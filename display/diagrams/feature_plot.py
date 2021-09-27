import matplotlib.pyplot as plt
import numpy as np
import os


def feature_plot(feature, i_label, i_values, j_label, j_values, results_folder, show, transparent):

    label_size = 16
    tick_size = 14

    # plt.title(feature)
    plt.pcolormesh(np.loadtxt(results_folder+'/'+feature+'.csv', delimiter=',').T)

    plt.xlabel(i_label, fontsize=label_size)
    plt.ylabel(j_label, fontsize=label_size)

    positions = np.linspace(0, len(i_values), 4)
    ticks = np.round(np.linspace(min(i_values), max(i_values), 4), 2)
    plt.xticks(positions, ticks)

    positions = np.linspace(0, len(j_values), 4)
    ticks = np.round(np.linspace(min(j_values), max(j_values), 4), 2)
    plt.yticks(positions, ticks)

    plt.tick_params(axis='both', which='major', labelsize=tick_size)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=tick_size)

    plt.tight_layout()

    if results_folder is not None:
        features_folder = os.path.join(results_folder, 'Plots', 'Features')
        os.makedirs(features_folder, exist_ok=True)
        feature_file = os.path.join(features_folder, feature+'.png')
        plt.savefig(feature_file, transparent=transparent)
    if show:
        plt.show()
    plt.close()
