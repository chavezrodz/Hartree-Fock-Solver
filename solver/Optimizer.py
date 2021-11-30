import itertools
import numpy as np
import os
import utils as utils


def fill_nans_nearest(arr):
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    return arr[np.arange(idx.shape[0])[:, None], idx]


def Optimizer_touchup(MFPs, Convergence_Grid):
    """
    Input list of arrays of energy across phase region,
    return best guess per region
    """
    # Indices where convergence failed
    indices = np.where(Convergence_Grid == 0,)
    indices = np.transpose(np.stack(indices))
    indices = list(map(tuple, indices))

    # Replace unconverged with Nans
    for v in indices:
        MFPs[v] = np.nan

    # Replace Nans with nearest neighbours
    for i in range(5):
        MFPs[:, :, i] = fill_nans_nearest(MFPs[:, :, i])
    return MFPs


def Optimizer_smoothing(mfps, sigma=[1, 1]):
    y = np.zeros(mfps.shape)
    for i in range(mfps.shape[2]):
        y[:, :, i] = sp.ndimage.filters.gaussian_filter(mfps[:, :, i], sigma, mode='nearest')
    return y


def Optimizer_exhaustive(input_folder, trials_list, input_MFP=False, verbose=False):
    print("Starting Optimizer")
    """
    Input list of arrays of energy across phase region,
    return best guess per region
    """

    n_mfps = len(trials_list[0])

    folderlist = ['Guess'+str(MF_params)
                  for MF_params in np.array(trials_list)]

    E_Tower, C_Tower = utils.load_energies_conv(input_folder, folderlist)
    diag_shape = E_Tower.shape[:-1]

    # Find Indices of lowest energies across stack
    ind = np.argmin(E_Tower, axis=-1)
    # Lowest achievable energy
    Optimal_Energy = np.take_along_axis(E_Tower, np.expand_dims(ind, axis=-1), axis=-1)
    Optimal_Energy = np.squeeze(Optimal_Energy)
    Optimal_Convergence = np.take_along_axis(C_Tower, np.expand_dims(ind, axis=-1), axis=-1)
    Optimal_Convergence = np.squeeze(Optimal_Convergence)

    if input_MFP:
        Solutions = utils.load_solutions(input_folder, folderlist)
        Unconverged_Sols = np.empty(Solutions.shape)
        Unconverged_Sols[:] = np.nan

        # Solutions and states
        # Optimal_States = np.take_along_axis(States, np.expand_dims(ind, axis=-1), axis=-1)
        # Optimal_States = np.squeeze(Optimal_States)

        Optimal_Solutions = np.zeros((*diag_shape, n_mfps))

        i, j = np.indices(diag_shape, sparse=True)
        i, j = i.flatten(), j.flatten()

        for v in itertools.product(i, j):
            Optimal_Solutions[v] = Solutions[v][ind[v]]

        Optimal_Guesses = Optimal_Solutions

    else:
        # Recover best guess across phase diagram
        Optimal_Guesses = np.zeros((*diag_shape, n_mfps))
        i, j = np.indices(diag_shape, sparse=True)
        i, j = i.flatten(), j.flatten()

        for v in itertools.product(i, j):
            Optimal_Guesses[v] = np.array(trials_list[ind[v]])
            if verbose:
                print('i ind:', v[0], 'j ind:', v[1], 'Best guess:', trials_list[ind[v]])
    return Optimal_Guesses, Optimal_Energy
