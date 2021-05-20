import matplotlib.pyplot as plt
import numpy as np
import os


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
