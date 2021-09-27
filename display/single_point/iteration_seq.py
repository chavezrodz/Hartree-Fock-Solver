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
    axs[1].set_ylabel(r'Mixing Factor $\alpha_{mix}$', fontsize=16)
    plt.xlabel('Iteration')
    if results_folder is not None:
        plt.savefig(results_folder+'/Iteration_sequence.png')
    if show:
        plt.show()
    plt.close()


def Iteration_comparison(Solver_fixed, Solver_scheduled):
    title_font = 14
    label_font = 16
    leg_font = 14
    tick_size = 14

    label_dict = Solver_fixed.Hamiltonian.dict_symbol

    fig, axs = plt.subplots(3, sharex=True)
    steps = Solver_fixed.sol_seq.shape[0]

    ax = axs[0]
    for i in range(Solver_fixed.sol_seq.shape[1]):
        ax.plot(np.arange(steps), Solver_fixed.sol_seq[:, i], label=label_dict[i])
    ax.set_title('Fixed Mixing Rate = {}'.format(Solver_fixed.beta), fontsize=title_font)
    ax.set_xlim(0, 50)
    ax.set_ylabel('Mean Field\n Parameters', fontsize=label_font)

    ax.text(
        1, 1.15*Solver_fixed.sol_seq[:].max(),
        'a)',
        ha='left', va='bottom', wrap=True, fontsize=14)

    ax.set_facecolor("white")
    ax.grid(b=True, color='black', linewidth=0.4)
    ax.tick_params(axis='both', which='both', labelsize=tick_size)
    for spine in ax.spines.values():
        spine.set_color('0.3')

    ax = axs[1]
    steps = Solver_scheduled.sol_seq.shape[0]
    for i in range(Solver_fixed.sol_seq.shape[1]):
        ax.plot(np.arange(steps), Solver_scheduled.sol_seq[:, i], label=label_dict[i])
    ax.set_title('Scheduled Mixing rate', fontsize=title_font)
    ax.set_xlim(0, 50)
    ax.set_ylabel('Mean Field\n Parameters', fontsize=label_font)

    ax.text(
        1, 1.15*Solver_scheduled.sol_seq[:].max(),
        'b)',
        ha='left', va='bottom', wrap=True, fontsize=14)

    ax.legend(fontsize=leg_font, ncol=2)

    ax.set_facecolor("white")
    ax.tick_params(axis='both', which='both', labelsize=tick_size)
    ax.grid(b=True, color='grey', linewidth=0.3)
    for spine in ax.spines.values():
        spine.set_color('0.3')

    ax = axs[2]
    steps = len(Solver_scheduled.beta_seq)
    ax.scatter(np.arange(steps), Solver_scheduled.beta_seq)
    ax.set_xlim(0, 50)
    ax.set_ylabel('Mixing Factor\n' + r'$\alpha_{mix}$', fontsize=label_font)

    ax.text(
        1, 1.05*Solver_scheduled.beta_seq.max(),
        'c)',
        ha='left', va='bottom', wrap=True, fontsize=14)

    ax.set_facecolor("white")
    ax.tick_params(axis='both', which='both', labelsize=tick_size)
    ax.grid(b=True, color='grey', linewidth=0.3)
    for spine in ax.spines.values():
        spine.set_color('0.3')

    plt.tight_layout()
    plt.tick_params(axis='both', which='major', labelsize=tick_size)
    plt.xlabel('Iteration', fontsize=title_font)
    fig.savefig('FixedVsScheduled.png', bbox_inches='tight')
    print('scheduled mixing rate, converged in {} steps'.format(Solver_scheduled.count))
    plt.show()
