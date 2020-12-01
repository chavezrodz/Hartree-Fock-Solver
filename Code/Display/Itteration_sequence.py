from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.interpolate
import numpy as np
import glob
import os

import seaborn as sns
sns.set_theme(font_scale=3)
sns.set_context("paper", font_scale=3)


def Itteration_sequence(Solver):
    fig, axs = plt.subplots(2)
    fig.suptitle('Converged: '+str(Solver.converged))
    for i in range(Solver.N_params):
        axs[0].plot(np.arange(Solver.sol_seq.shape[0]), Solver.sol_seq[:, i], label=Solver.Hamiltonian.Dict[i])
    axs[0].set_title('Mean Field Parameters')
    axs[0].legend()

    axs[1].scatter(np.arange(len(Solver.beta_seq)), Solver.beta_seq)
    axs[1].set_title('Mixing Factor')
    plt.xlabel('Itteration')
    plt.show()


def Itteration_comparison(Solver_fixed, Solver_scheduled):
    fig, axs = plt.subplots(3)
    for i in range(Solver_fixed.N_params):
        axs[0].plot(np.arange(Solver_fixed.sol_seq.shape[0]), Solver_fixed.sol_seq[:, i], label=Solver_fixed.Hamiltonian.Dict[i])
    axs[0].set_title('Mean Field Parameters, Fixed Mixing Rate = {}, unconverged'.format(Solver_fixed.beta))
    axs[0].set_xlim(0, 50)
    axs[0].legend()

    for i in range(Solver_scheduled.N_params):
        axs[1].plot(np.arange(Solver_scheduled.sol_seq.shape[0]), Solver_scheduled.sol_seq[:, i], label=Solver_scheduled.Hamiltonian.Dict[i])
    axs[1].set_title('Mean Field Parameters, scheduled mixing rate, converged in {} steps'.format(Solver_scheduled.count))
    axs[1].set_xlim(0, 50)
    axs[1].legend()

    axs[2].scatter(np.arange(len(Solver_scheduled.beta_seq)), Solver_scheduled.beta_seq)
    axs[2].set_xlim(0, 50)
    axs[2].set_title('Mixing Factor')
    plt.xlabel('Itteration')
    plt.tight_layout()
    plt.savefig('FixedVsScheduled.png', transparent=False)
    plt.show()