from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.interpolate
import numpy as np
import glob
import os

import seaborn as sns
sns.set_theme(font_scale=12)
sns.set_context("paper", font_scale=12)


def Iteration_sequence(Solver, results_folder=None, show=True):
    fig, axs = plt.subplots(2)
    fig.suptitle('Converged: '+str(Solver.converged))
    for i in range(len(Solver.Hamiltonian.MF_params)):
        axs[0].plot(np.arange(Solver.sol_seq.shape[0]), Solver.sol_seq[:, i], label=Solver.Hamiltonian.Dict[i])
    axs[0].set_title('Mean Field Parameters')
    axs[0].legend()

    axs[1].scatter(np.arange(len(Solver.beta_seq)), Solver.beta_seq)
    axs[1].set_title('Mixing Factor')
    plt.xlabel('Iteration')
    if results_folder is not None:
        plt.savefig(results_folder+'/Iteration_sequence.png')
    if show:
        plt.show()
    plt.close()


def Iteration_comparison(Solver_fixed, Solver_scheduled):
    title_font = 12
    leg_font = 10
    fig, axs = plt.subplots(3, sharex=True)
    steps = Solver_fixed.sol_seq.shape[0]
    for i in range(Solver_fixed.sol_seq.shape[1]):
        axs[0].plot(np.arange(steps), Solver_fixed.sol_seq[:, i], label=Solver_fixed.Hamiltonian.Dict[i])
    axs[0].set_title('Mean Field Parameters, Fixed Mixing Rate = {}, unconverged'.format(Solver_fixed.beta), fontsize=title_font)
    axs[0].set_xlim(0, 50)
    axs[0].tick_params(axis='both', which='major', labelsize=12)

    steps = Solver_scheduled.sol_seq.shape[0]
    for i in range(Solver_fixed.sol_seq.shape[1]):
        axs[1].plot(np.arange(steps), Solver_scheduled.sol_seq[:, i], label=Solver_scheduled.Hamiltonian.Dict[i])
    axs[1].set_title('Mean Field Parameters, scheduled mixing rate, converged in {} steps'.format(Solver_scheduled.count), fontsize=title_font)
    axs[1].set_xlim(0, 50)
    axs[1].legend(fontsize=leg_font)
    axs[1].tick_params(axis='both', which='major', labelsize=12)

    steps = len(Solver_scheduled.beta_seq)
    axs[2].scatter(np.arange(steps), Solver_scheduled.beta_seq)
    axs[2].set_xlim(0, 50)
    axs[2].set_title('Mixing Factor', fontsize=title_font)

    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.xlabel('Iteration', fontsize=title_font)
    plt.tight_layout()
    plt.savefig('FixedVsScheduled.png', bbox_inches='tight')
    plt.show()
